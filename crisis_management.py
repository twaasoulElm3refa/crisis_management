import os
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# ---- DB helpers (already implemented elsewhere) ----
from database import fetch_latest_result, save_result

# -----------------------------------------------------------------------------
# OpenAI setup
# -----------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    date:OPtional[date]=None

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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_language(d: Dict[str, Any]) -> None:
    """Map Arabic/English labels to ar/en to satisfy prompt rule."""
    lang = (d.get("language") or "").strip().lower()
    if lang in ("العربية", "arabic", "ar"):
        d["language"] = "ar"
    elif lang in ("الإنجليزية", "english", "en"):
        d["language"] = "en"

def _to_llm_input(data: Optional[CrisisInput], data_raw: Optional[str]):
    """Prefer structured dict; fall back to raw string; else {}."""
    if data is not None:
        d = data.model_dump(exclude_none=True)
        if isinstance(d, dict):
            _normalize_language(d)
        return d
    elif data_raw:
        return data_raw
    return {}

# -----------------------------------------------------------------------------
# Core LLM function (your exact logic, just wrapped safely)
# -----------------------------------------------------------------------------
def crisis_management_update3(data: Any) -> str:
    """
    Calls OpenAI with your structured Arabic prompt and returns the raw model output.
    NOTE: We expect the model to return ONE valid JSON object (UTF-8).
    """
    prompt = f''' أنت مستشار أزمات اتصالية احترافي.
        التزم بالقانون والسياسات الداخلية، وتجنّب الافتراضات غير المؤكدة.
        إن لم تتوفر معلومة، أوصِ بجمعها بدل تخمينها.
        قدم التفسير مع ذكر كيفية التنفيذ بالتفصيل لكل نقطة وذكر سبب الاختيار لها.
        اكتب بالعربية الفصحى أو الإنجليزية بحسب لغة المدخلات. قدّم مخرجات منظمة قابلة للتنفيذ فورًا  .
        التصنيفات (القيم الممكنة):

         تصنيف الأزمة: السمعة، تشغيلية، قانونية، سلامة، اختراق بيانات، معلومات مضللة، اجتماعية/أخلاقية ( مع ذكر السبب)
        مستوى المخاطر: منخفض، متوسط، مرتفع، حرج ( مع ذكر السبب)
        الاستراتيجية: إقرار وتفسير، إقرار والتحقيق، احتواء وتهدئة، تصحيح المعلومات، نفي مدعوم بالأدلة، اعتذار مشروط، اعتذار كامل، مراقبة صامتة ( مع ذكر السبب)
        
        حساب المخاطر:
        الحدود: الوصول ∈ [0..20]، السرعة ∈ [0..15]، السلبية ∈ [0..15]، السلامة ∈ [0..20]، القانون ∈ [0..10]، حساسية الشخصيات ∈ [0..10]، الموثوقية ∈ [0..10]
        المعادلة: درجة_المخاطر = الوصول + السرعة + السلبية + السلامة + القانون + حساسية_الشخصيات + الموثوقية (من 0 إلى 100)
        تصنيف المستوى بحسب الدرجة: 0–29 منخفض، 30–59 متوسط، 60–79 مرتفع، 80–100 حرج
        عند نقص البيانات: تُذكر ضمن «النواقص المطلوبة» بدل التخمين.
        قواعد الاختيار:
        إذا كانت تبعات السلامة = صحيح → تُعطى أولوية لقوالب السلامة مع التصعيد القانوني.
        إذا كانت الحساسية القانونية = مرتفعة/حرجة → صياغة شديدة الحذر + مراجعة قانونية إلزامية + تجنّب التفاصيل غير المثبتة.
        إذا كان تصنيف الأزمة = معلومات مضللة مع أدلّة قوية → تصحيح المعلومات أو نفي مدعوم بالأدلة.
        إذا وُجد مؤشر واضح على مسؤولية داخلية → إقرار وتفسير أو إقرار والتحقيق (وقد يُضاف اعتذار مشروط/كامل وفق الأدلة).

         (دون أي string خارجي أو ) وفق المخطط التالي (نفس المفاتيح
         "تصنيف_الأزمة": ["<من القائمة أعلاه>"],
         خوارزمية تقييم المخاطر مع التفسير (قابلة للتخصيص) مع التفسير لكل نقطة 
            نحسب معدل الخطر من 100 وفق أوزان قابلة للضبط:
            الانتشار Reach (R): 0–20 (حجم الذكر).
            السرعة Velocity (V): 0–15 (معدل الزيادة).
            السلبية Sentiment (S): 0–15 (قطبية/حدة).
            السلامة Safety (H): 0–20 (تهديد لحياة/صحة؟).
            القانون Legal (L): 0–10 (احتمال تبعات قانونية).
            VIP/حساسية سياسة (P): 0–10 (شخصيات مهمة/رموز؟).
            الموثوقية Evidence (E): 0–10 (مدعوم بأدلة؟ كلما زادت الموثوقية السلبية ارتفع الخطر).
            الصيغة:
            معدل الخطر = R + V + S + H + L + P + E   // من 0 إلى 100

             مع التفسير العتبات الافتراضية:
            0–29 = منخفض
            30–59 = متوسط
            60–79 = عالي
            80–100 = طارئ
            > تُستمد قيم R/V/S تلقائيًا من تحليلات خارجية (إن توفرت) أو من تقدير المستخدم إذا لم تتكاملوا مع مصادر رصد.

            3. حساب المخاطر:
            استدعاء risk_scorer.calculate(payload) لإنتاج معدل الخطر وrisk_level.
            ---

             منطق الاختيار الاستراتيجي (Rules قبل الـLLM)مع التفسير المفصل  

            إذا safety_implications = true → تفعيل قوالب السلامة أولًا (تحذير/إجراءات فورية) + تصعيد قانوني.
            إذا legal_sensitivity = High → صياغة حذرة جدًا + مراجعة قانونية إلزامية + تجنب تفاصيل غير مثبتة.
            إذا crisis_category = MISINFORMATION وevidence strong → CORRECT_INFO أو EVIDENCE_BASED_DENIAL.
            إذا توفرت قرائن مسؤولية داخلية واضحة → ACK_EXPLAIN أو ACK_INVESTIGATE وربما اعتذار مشروط/كامل.

            > حاجز أمان: لا يعتمد على النفي إلّا مع مبررات/أدلة قابلة للإبراز.

          "خطة_العمل": 
            "الأفق_الزمني_بالساعات": <عدد_صحيح>,
            "المراحل": 
                "النافذة": "... ساعة",
                "الإجراءات":  "المهمة":, "المسؤول": , "سقف_زمني": "<مدة>", "ملاحظات":  ,
            "المسؤولون": ,
            "أصول_الاتصال":
              "بيان": 
                "النوع":
                "القناة": ,
                "المحتوى": "",
                "التوقيع": "<string>"
            "السوشال ميديا و الوسائل المستهدفة ":
              "مطلوب": [],
              "مستبعد": [],
            "المتابعة": 
              "التواتر": "<كل_ساعة|كل_4_ساعات|يوميًا>",
              "المقاييس>": "<تغير_الحجم>", "<تحول_المشاعر>", "<متوسط_زمن_الاستجابة>" ,
              "مؤشرات_الأداء": 
             خلال الوقت المذكور
                "متوسط_زمن_الاستجابة_اجتماعيًا": "<قيمة زمنية>",
                "معدل_الحل_خلال_حدد عدد_الساعات حسب_المشكلة": "<قيمة أو نسبة>",
              "محفزات_التصعيد": ["<string>"],
              "مراجع_قاعدة_المعرفة": ["<string>"],
              "سجل_التدقيق": طابع_زمني :{data.date}, "حدث": "<string>", "بواسطة": "<نظام|مستخدم>" 
                  "موجز ": "<string>" يوضح الخطوات المطلوبة بشكل مقالي واضح ودقيق 

                قواعد صارمة للإخراج:
                - إذا اللغة = "ar" اجعل human_summary بالعربية الفصحى، وإلا بالإنجليزية.
                - املأ "gaps_required" بأي معلومات تنقصك بدلاً من التخمين.
                - لا تضف حقولًا غير معرفة، ولا تغيّر أسماء الحقول.'''

    # Build the user message as REAL JSON (or a string block)
    if isinstance(data, (dict, list)):
        data_block = json.dumps(data, ensure_ascii=False)
    else:
        data_block = str(data)
    user_msg = "data:\n" + data_block

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "system","content": "You are an expert assistant. Always give long, detailed, and accurate answers with examples."},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# Ensure the model output is ONE valid JSON object (stringified)
# -----------------------------------------------------------------------------
def normalize_result_to_json_string(raw: str) -> str:
    """
    Accepts the LLM raw string, strips code fences if any,
    verifies it's valid JSON (object or array), and returns a UTF-8 string.
    If it's not valid JSON, raises ValueError.
    """
    s = (raw or "").strip()

    # Strip code-fence wrappers if present
    if s.startswith("```"):
        s = s.strip("`")
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) == 2 else parts[0]
        s = s.strip()

    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}")

    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

# -----------------------------------------------------------------------------
# FastAPI app & CORS
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Crisis Management API", version="1.0.0")

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
    """Run the job in background, store either result or error JSON."""
    try:
        data_for_llm = _to_llm_input(payload.data, payload.data_raw)
        raw = crisis_management_update3(data_for_llm)
        normalized = normalize_result_to_json_string(raw)
        save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=normalized)
    except Exception as e:
        err_json = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=err_json)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start", response_model=ApiStatus)
def start(payload: StartPayload, bg: BackgroundTasks):
    """
    Async mode: enqueue background generation and return 'processing' immediately.
    WP polls /result with request_id.
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
        raw = crisis_management_update3(data_for_llm)
        normalized = normalize_result_to_json_string(raw)
        save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=normalized)
        return ApiStatus(status="done", result=normalized)
    except Exception as e:
        return ApiStatus(status="error", message=f"{type(e).__name__}: {e}")

@app.post("/result", response_model=ApiStatus)
def get_result(req: ResultRequest):
    """
    Polling endpoint: returns 'processing' or 'done' + stored result (JSON string).
    """
    row = fetch_latest_result(req.request_id)
    if not row:
        return ApiStatus(status="processing")
    return ApiStatus(status="done", result=row["edited_result"] or row["result"])





