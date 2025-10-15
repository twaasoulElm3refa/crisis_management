import os
import time
import uuid
from datetime import date as DtDate
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import jwt  # PyJWT

# ---- DB helpers (already implemented elsewhere) ----
from database import fetch_latest_result, save_result

# -----------------------------------------------------------------------------
# OpenAI setup
# -----------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------------------------
# Security / JWT
# -----------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET") or os.urandom(32)
JWT_ALG = "HS256"

def _make_jwt(session_id: str, user_id: int) -> str:
    payload = {
        "sid": session_id,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 2,  # 2 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def _verify_jwt(bearer: Optional[str]):
    if not bearer or not bearer.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = bearer.split(" ", 1)[1]
    try:
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -----------------------------------------------------------------------------
# Pydantic models (input & output envelopes)
# -----------------------------------------------------------------------------
class CrisisInput(BaseModel):
    crisis_description: Optional[str] = None
    sector: Optional[str] = None
    origin: Optional[str] = None
    audience_locales: Optional[List[str]] = None
    public_sentiment: Optional[List[str]] = None
    urgency_level: Optional[str] = None
    language: Optional[str] = None
    preferred_tone: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    brand_style: Optional[Dict[str, Any]] = None
    kb_tags: Optional[List[str]] = None
    channels_context: Optional[Dict[str, Any]] = None
    time_horizon_hours: Optional[int] = None
    coverage: Optional[str] = None
    legal_sensitivity: Optional[str] = None
    safety_implications: Optional[bool] = None
    vip_involved: Optional[bool] = None
    date: Optional[DtDate] = None  # optional "today" if you send it from PHP

class StartPayload(BaseModel):
    request_id: int = Field(..., gt=0)
    user_id: int    = Field(..., gt=0)
    data: Optional[CrisisInput] = None
    data_raw: Optional[str] = None

class ResultRequest(BaseModel):
    request_id: int = Field(..., gt=0)

class ApiStatus(BaseModel):
    status: str
    result: Optional[str] = None
    message: Optional[str] = None

# ==== NEW: Chat models to match the WordPress plugin ====

class SessionIn(BaseModel):
    user_id: int
    wp_nonce: Optional[str] = None  # carried through but not used here

class SessionOut(BaseModel):
    session_id: str
    token: str

class VisibleValue(BaseModel):
    # From your WP visible_values (most fields optional)
    id: Optional[int] = None
    crisis_description: Optional[str] = None
    sector: Optional[str] = None
    origin: Optional[str] = None
    audience_locales: Optional[str] = None  # plugin sends comma-joined for display, OK as string
    public_sentiment: Optional[str] = None
    urgency_level: Optional[str] = None
    language: Optional[str] = None
    preferred_tone: Optional[str] = None
    constraints: Optional[str] = None
    kb_tags: Optional[str] = None
    date: Optional[str] = None

    # critical for chat context
    article: Optional[str] = None  # latest narrative text (edited or raw), from localStorage/DB

class ChatIn(BaseModel):
    session_id: str
    user_id: int
    message: str
    visible_values: List[VisibleValue] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_language(d: Dict[str, Any]) -> None:
    """Map Arabic/English labels to ar/en."""
    lang = (d.get("language") or "").strip().lower()
    if lang in ("العربية", "arabic", "ar"):
        d["language"] = "ar"
    elif lang in ("الإنجليزية", "english", "en"):
        d["language"] = "en"

def _to_llm_input(data: Optional[CrisisInput], data_raw: Optional[str]):
    """Prefer structured dict; otherwise raw string; else {}."""
    if data is not None:
        d = data.model_dump(exclude_none=True)
        if isinstance(d, dict):
            _normalize_language(d)
        return d
    elif data_raw:
        return data_raw
    return {}

def _values_to_context(values: List[VisibleValue]) -> str:
    """Arabic context line(s) for chat based on latest visible values."""
    if not values:
        return "لا توجد بيانات مرئية حالياً لهذا المستخدم."
    v = values[0]
    pieces = []
    if v.sector:             pieces.append(f"القطاع: {v.sector}")
    if v.origin:             pieces.append(f"المصدر: {v.origin}")
    if v.language:           pieces.append(f"اللغة: {v.language}")
    if v.urgency_level:      pieces.append(f"الإلحاح: {v.urgency_level}")
    if v.preferred_tone:     pieces.append(f"النبرة: {v.preferred_tone}")
    if v.kb_tags:            pieces.append(f"وسوم: {v.kb_tags}")
    if v.constraints:        pieces.append(f"قيود: {v.constraints}")
    if v.audience_locales:   pieces.append(f"مناطق الجمهور: {v.audience_locales}")
    if v.public_sentiment:   pieces.append(f"انطباع الجمهور: {v.public_sentiment}")
    if v.date:               pieces.append(f"التاريخ: {v.date}")
    if v.crisis_description: pieces.append(f"وصف الأزمة: {v.crisis_description}")
    # Include latest narrative text if available
    if v.article:
        # keep it last, labelled clearly
        pieces.append(f"أحدث نص (تحليلي/صياغي):\n{v.article}")
    return " | ".join(pieces) if pieces else "لا توجد تفاصيل كافية."

# -----------------------------------------------------------------------------
# Core LLM function — narrative (no JSON)
# -----------------------------------------------------------------------------
def crisis_management_narrative(data: Any) -> str:
    """
    Returns a HUMAN narrative report (no JSON). Arabic or English based on input.
    """
    prompt = f"""
    أنت مستشار أزمات اتصالية احترافي.
    - التزم بالقانون والسياسات الداخلية، وتجنّب الافتراضات غير المؤكدة.
    - إن لم تتوفر معلومة، أوصِ بجمعها بدل تخمينها.
    - اكتب بنفس لغة المدخلات (ar أو en). إذا كانت اللغة = ar فاستخدم العربية الفصحى، وإلا فاستخدم الإنجليزية.
    - لا تُخرج JSON أو جداول بتنسيق برمجي أو أسوار تعليمية؛ اكتب نصًا عاديًا منسقًا بعناوين ونقاط.
    - اذكر دوماً «السبب» وراء كل تصنيف أو قرار أو قناة أو نبرة أو إجراء.
    - احترم قيود العلامة (forbidden_terms) ولا تستخدمها في الصياغة، واستخدم التوقيع إن وُجد.
    - احترم سياق القنوات: أدرج القنوات المطلوبة واستبعد القنوات المحددة، مع تبرير الاختيار.
    - عند نقص البيانات، أدرجها في قسم «نواقص مطلوبة» واقترح آلية جمعها (مصدر، مسؤول، مهلة).

    استخدم البنية التالية كمقال عملي قابل للتنفيذ (عناوين واضحة ونقاط مرقّمة حيث يلزم):

    1) الموجز التنفيذي
       قدم 3–5 جمل تلخص الوضع، المخاطر المباشرة، وما تنوي فعله الآن.

    2) تشخيص الأزمة (مع السبب)
       - صنّف نوع/أنواع الأزمة من القائمة: السمعة، تشغيلية، قانونية، سلامة، اختراق بيانات، معلومات مضللة، اجتماعية/أخلاقية.
       - فسّر لماذا وقع الاختيار على كل تصنيف، بالاستناد إلى المعطيات المتاحة.

    3) تقييم المخاطر الكمي (مع التفسير)
       وضّح النقاط التالية كقائمة بعناصر مُسمّاة، مع الدرجة والسبب لكل عنصر:
       - Reach (R) 0–20: حجم الذكر/الانتشار ولماذا.
       - Velocity (V) 0–15: سرعة التصاعد ولماذا.
       - Sentiment (S) 0–15: السلبية/الحدة ولماذا.
       - Safety (H) 0–20: تبعات السلامة ولماذا.
       - Legal (L) 0–10: الحساسية القانونية ولماذا.
       - VIP/Policy (P) 0–10: حساسية الشخصيات/الرموز ولماذا.
       - Evidence (E) 0–10: قوة الأدلة السلبية ولماذا.
       ثم احسب «معدل الخطر = R + V + S + H + L + P + E» (0–100)، واصفاً «مستوى الخطر» وفق العتبات:
       0–29 منخفض، 30–59 متوسط، 60–79 مرتفع، 80–100 حرج. اشرح سبب المستوى النهائي.

    4) الاستراتيجية المختارة والنبرة (مع السبب)
       اختر من: إقرار وتفسير، إقرار والتحقيق، احتواء وتهدئة، تصحيح المعلومات، نفي مدعوم بالأدلة، اعتذار مشروط، اعتذار كامل، مراقبة صامتة.
       - فسّر لماذا هذه الاستراتيجية مناسبة لهذه الحالة.
       - حدد «النبرة» (رسمية، مهنية، إنسانية، مطمئنة، حازمة) ولماذا.

    5) خطة العمل الزمنية (عملياتيًا)
       استند إلى الأفق الزمني بالساعات إن توفّر. قسّم الخطة إلى مراحل زمنية قصيرة (مثلاً 0–6، 6–24، 24–48، 48–72 إن انطبق).
       في كل مرحلة قدّم عناصر قابلة للتنفيذ بصيغة:
       المهمة — المسؤول — المهلة (SLA) — ملاحظات/سبب مختصر.
       اجعل كل عنصر محددًا وقابلاً للقياس قدر الإمكان.

    6) القنوات والتكتيكات الإعلامية
       - القنوات المطلوبة: فسّر سبب اختيار كل قناة وكيف ستُستخدم.
       - القنوات المستبعدة: فسّر سبب الاستبعاد والمخاطر المتوقعة إن استُخدمت.
       التزم بقيود «سياق القنوات» الواردة في المدخلات.

    7) المتابعة والقياس
       - التواتر (كل ساعة/كل 4 ساعات/يوميًا) ولماذا يناسب الوضع.
       - المقاييس التي ستُرصد (مثل: تغير الحجم، تحوّل المشاعر، متوسط زمن الاستجابة) ولماذا هذه المقاييس مهمة.

    8) مؤشرات الأداء المستهدفة
       اذكر قيمًا أو نطاقات للمؤشرات الرئيسية (مثل: متوسط زمن الاستجابة اجتماعيًا، معدل الحل خلال الأفق الزمني)، وسبب اختيار هذه الحدود.

    9) بيان/نص اتصال مقترح
       قدّم مسودة موجزة بصياغة مهنية تراعي:
       - عدم تضمين أسماء/معلومات حساسة غير مؤكدة.
       - احترام forbidden_terms والتوقيع إن وُجد.
       - الانسجام مع الاستراتيجية والنبرة المختارتين.

    10) محفزات التصعيد
        اذكر الحالات التي تستدعي التصعيد (قانوني/تنفيذي/سلامة)، ولماذا.

    11) نواقص مطلوبة
        عدّد المعلومات غير المتوفرة التي تمنع دقة أعلى، واقترح طريقة جمعها (المصدر، المسؤول، الإطار الزمني).

    12) سجل موجز للتدقيق
        سطّر بنقاط مختصرة أحدث الإجراءات/القرارات (طابع زمني، الحدث، من قام به)، دون ذكر أسماء شخصية ما لم تكن جزءًا من السياق المصرّح به.

    قواعد القرار المسبقة التي يجب مراعاتها داخل التحليل:
    - إذا كانت تبعات السلامة = صحيح: أعطِ أولوية لقوالب السلامة مع تصعيد قانوني.
    - إذا كانت الحساسية القانونية = مرتفعة/حرجة: صياغة شديدة الحذر + مراجعة قانونية إلزامية + تجنّب التفاصيل غير المثبتة.
    - إذا كان النوع = معلومات مضللة مع أدلة قوية: اتجه لتصحيح المعلومات أو نفي مدعوم بالأدلة.
    - إذا وُجدت مؤشرات على مسؤولية داخلية: فضّل «إقرار وتفسير» أو «إقرار والتحقيق»، وقد تُضاف صيغة اعتذار مشروط/كامل وفق الأدلة.

    مهم:
    - كن عمليًا ودقيقًا، وقدّم سببًا واضحًا لكل اختيار.
    - لا تستخدم JSON أو ترميز برمجي؛ الإخراج نصي إنساني قابل للقراءة والتطبيق الفوري.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "system", "content": "You are an expert assistant. Always give long, detailed, and accurate answers with examples."},
            {"role": "user", "content": f" data:{data}"},
        ],
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# FastAPI app & CORS
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app = FastAPI(title="Crisis Management API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Background processor
# -----------------------------------------------------------------------------
def process_job(payload: StartPayload):
    """Run in background; store RAW narrative text (no JSON normalization)."""
    try:
        data_for_llm = _to_llm_input(payload.data, payload.data_raw)
        raw = crisis_management_narrative(data_for_llm)
        save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=raw)
    except Exception as e:
        err_text = f"ERROR: {type(e).__name__}: {e}"
        try:
            save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=err_text)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Routes (existing)
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start", response_model=ApiStatus)
def start(payload: StartPayload, bg: BackgroundTasks):
    """
    Async mode: enqueue job and return 'processing' immediately.
    """
    existing = fetch_latest_result(payload.request_id)
    if existing:
        return ApiStatus(status="done", result=existing["edited_result"] or existing["result"])

    bg.add_task(process_job, payload)
    return ApiStatus(status="processing")

@app.post("/start_sync", response_model=ApiStatus)
def start_sync(payload: StartPayload):
    """
    Sync mode: generate immediately and return 'done' + result.
    """
    existing = fetch_latest_result(payload.request_id)
    if existing:
        return ApiStatus(status="done", result=existing["edited_result"] or existing["result"])

    try:
        data_for_llm = _to_llm_input(payload.data, payload.data_raw)
        raw = crisis_management_narrative(data_for_llm)
        save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=raw)
        return ApiStatus(status="done", result=raw)
    except Exception as e:
        err_text = f"ERROR: {type(e).__name__}: {e}"
        try:
            save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=err_text)
        finally:
            return ApiStatus(status="error", message=err_text)

@app.post("/result", response_model=ApiStatus)
def get_result(req: ResultRequest):
    """
    Poll for the stored result text.
    """
    row = fetch_latest_result(req.request_id)
    if not row:
        return ApiStatus(status="processing")
    return ApiStatus(status="done", result=row["edited_result"] or row["result"])

# -----------------------------------------------------------------------------
# NEW: Chat routes
# -----------------------------------------------------------------------------
@app.post("/session", response_model=SessionOut)
def create_session(body: SessionIn):
    """
    Creates a short-lived JWT for chat. The token is required in Authorization header for /chat.
    """
    sid = str(uuid.uuid4())
    token = _make_jwt(sid, body.user_id)
    return SessionOut(session_id=sid, token=token)

@app.post("/chat")
def chat(body: ChatIn, authorization: Optional[str] = Header(None)):
    """
    Streaming chat using OpenAI; relies on the visible_values you send from WP.
    Returns 'text/plain' stream (no SSE framing), exactly how your JS expects it.
    """
    _verify_jwt(authorization)

    # Build system prompt from visible values (Arabic default; switch by 'language' if needed)
    context = _values_to_context(body.visible_values)
    sys_prompt = (
        "أنت مساعد موثوق يجيب بالاعتماد على البيانات المرئية الحالية للمستخدم. "
        "إذا كانت المعلومة غير متوفرة في البيانات المرئية فاذكر ذلك صراحةً "
        "واقترح ما يمكن فعله للحصول عليها.\n\n"
        f"البيانات المرئية الحالية:\n{context}"
    )

    user_msg = body.message or ""
    def stream():
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_msg}
                ],
                stream=True
            )
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            # Surface readable error to the user stream
            yield f"\n[خطأ: {type(e).__name__}] {e}"

    return StreamingResponse(stream(), media_type="text/plain")
