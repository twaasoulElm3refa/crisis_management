import os
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
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
# Core LLM function (your exact logic, just wrapped safely)
# -----------------------------------------------------------------------------
def crisis_management_update2(data: Any) -> str:
    """
    Calls OpenAI with your structured Arabic prompt and returns the raw model output.
    NOTE: We expect the model to return ONE valid JSON object (UTF-8).
    """
    prompt = ''' أنت مستشار أزمات اتصالية احترافي.
    التزم بالقانون والسياسات الداخلية، وتجنّب الافتراضات غير المؤكدة.
    إن لم تتوفر معلومة، أوصِ بجمعها بدل تخمينها.
    اكتب بالعربية الفصحى أو الإنجليزية بحسب لغة المدخلات. قدّم مخرجات منظمة قابلة للتنفيذ فورًا.
التصنيفات (القيم الممكنة):

تصنيف الأزمة: السمعة، تشغيلية، قانونية، سلامة، اختراق بيانات، معلومات مضللة، اجتماعية/أخلاقية
مستوى المخاطر: منخفض، متوسط، مرتفع، حرج
الاستراتيجية: إقرار وتفسير، إقرار والتحقيق، احتواء وتهدئة، تصحيح المعلومات، نفي مدعوم بالأدلة، اعتذار مشروط، اعتذار كامل، مراقبة صامتة
النبرة: رسمية، مهنية، إنسانية، مطمئنة، حازمة
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

     (دون أي string خارجي أو Markdown) وفق المخطط التالي (نفس المفاتيح والترتيب إن أمكن):

    {{
  "وسوم_قاعدة_المعرفة": ["<string>"],
  "سياق_القنوات": {{
    "مطلوب": ["<string>"],
    "مستبعد": ["<string>"]
  }},
  "أفق_الزمن_بالساعات": <عدد_صحيح>,
  "نطاق_التغطية": "<داخلي|خارجي|داخلي_و_خارجي>",
  "الحساسية_القانونية": "<منخفضة|متوسطة|مرتفعة|حرجة>",
  "تبعات_السلامة": <صحيح|خطأ>,
  "وجود_شخصية_هامة": <صحيح|خطأ>,

  "التصنيف": {{
    "تصنيف_الأزمة": ["<من القائمة أعلاه>"],
    "مستوى_المخاطر": "<من القائمة أعلاه>",
    "تفصيل_درجة_المخاطر": {{
      "الوصول": <0-20>,
      "السرعة": <0-15>,
      "السلبية": <0-15>,
      "السلامة": <0-20>,
      "القانون": <0-10>,
      "حساسية_الشخصيات": <0-10>,
      "الموثوقية": <0-10>,
      "المجموع": <0-100>
    }},
    "الاستراتيجية": ["<من القائمة أعلاه>"],
    "النبرة": ["<من القائمة أعلاه>"]
  }},

  "القواعد_المطبقة": [
    {{ "الشرط": "<string>", "الأثر": "<string>" }}
  ],

  "النواقص_المطلوبة": ["<string>"],

  "ضوابط_وخصوصية": {{
    "خصوصية": true,
    "عدم_ذكر_أسماء": true,
    "مصطلحات_محظورة": ["<string>"]
  }}

  "خطة_العمل": {{
    "الأفق_الزمني_بالساعات": <عدد_صحيح>,
    "المراحل": [
      {{
        "النافذة": "0–6 ساعة",
        "الإجراءات": [
          {{ "المهمة": "<string>", "المسؤول": "<دور>", "سقف_زمني": "<مدة>", "ملاحظات": "<string اختياري>" }}
        ]
      }},
      {{ "النافذة": "6–24 ساعة", "الإجراءات": [] }},
      {{ "النافذة": "24–48 ساعة", "الإجراءات": [] }},
      {{ "النافذة": "48–72 ساعة", "الإجراءات": [] }}
    ],
    "المسؤولون": {{
      "قائد_العلاقات_العامة": "<string>",
      "المستشار_القانوني": "<string>",
      "مدير_الفرع": "<string>",
      "فريق_الرصد_الاجتماعي": "<string>"
    }},
    "أصول_الاتصال": {{
      "بيان": {{
        "النوع": "بيان صحفي",
        "القناة": "نشرة_صحفية",
        "المحتوى": "<string>",
        "التوقيع": "<string>"
      }}
    }},
    "القنوات": {{
      "مطلوب": ["<string>"],
      "مستبعد": ["<string>"]
    }},
    "المتابعة": {{
      "التواتر": "<كل_ساعة|كل_4_ساعات|يوميًا>",
      "المقاييس": ["<تغير_الحجم>", "<تحول_المشاعر>", "<متوسط_زمن_الاستجابة>"]
    }}
  }},

  "مؤشرات_الأداء": {{
    "تغير_الحجم_خلال_24_ساعة": "<قيمة أو نسبة>",
    "تحول_المشاعر_خلال_48_ساعة": "<قيمة أو نسبة>",
    "متوسط_زمن_الاستجابة_اجتماعيًا": "<قيمة زمنية>",
    "معدل_الحل_خلال_72_ساعة": "<قيمة أو نسبة>"
  }},

  "محفزات_التصعيد": ["<string>"],
  "مراجع_قاعدة_المعرفة": ["<string>"],
  "سجل_التدقيق": [
    {{ "طابع_زمني": "<تاريخ/وقت معياري>", "حدث": "<string>", "بواسطة": "<نظام|مستخدم>" }}
    ],
}}

      "موجز ": "<string>"
    }}

    قواعد صارمة للإخراج:
    - أعد كائن JSON واحد صالح نحويًا (UTF-8).
    - لا تُخرج أي أسطر تفسيرية أو Markdown أو تعليقات.
    - إذا اللغة = "ar" اجعل human_summary بالعربية الفصحى، وإلا بالإنجليزية.
    - املأ "gaps_required" بأي معلومات تنقصك بدلاً من التخمين.
    - لا تضف حقولًا غير معرفة، ولا تغيّر أسماء الحقول.
    '''
    user_msg = f"data: {data}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


# -----------------------------------------------------------------------------
# Small helper: ensure the model output is ONE valid JSON object (stringified)
# -----------------------------------------------------------------------------
def normalize_result_to_json_string(raw: str) -> str:
    """
    Accepts the LLM raw string, strips code fences if any,
    verifies it's valid JSON (object or array), and returns a UTF-8 string.
    If it's not valid JSON, raises ValueError.
    """
    s = (raw or "").strip()

    # Strip common code-fence wrappers if present
    if s.startswith("```"):
        # remove the first fence line and trailing fence
        s = s.strip("`")
        # heuristic: split by first newline after possible language tag
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) == 2 else parts[0]
        s = s.strip()

    # Try parse
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}")

    # Re-serialize with UTF-8 friendly settings (no ASCII escaping)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


# -----------------------------------------------------------------------------
# FastAPI app & schemas
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
from typing import Optional, List

class CrisisInput(BaseModel):
    crisis_description: Optional[str] = None
    sector: Optional[str] = None
    origin: Optional[str] = None
    audience_locales: Optional[List[str]] = None
    public_sentiment: Optional[str] = None
    urgency_level: Optional[str] = None
    language: Optional[str] = None
    preferred_tone: Optional[str] = None
    constraints: Optional[List[str]] = None
    brand_style: Optional[Dict[str, Any]] = None
    kb_tags: Optional[List[str]] = None
    channels_context: Optional[Dict[str, Any]] = None
    time_horizon_hours: Optional[int] = None
    coverage: Optional[str] = None
    legal_sensitivity: Optional[str] = None
    safety_implications: Optional[bool] = None
    vip_involved: Optional[bool] = None

class StartPayload(BaseModel):
    request_id: int = Field(..., gt=0)
    user_id: int    = Field(..., gt=0)
    data: Optional[CrisisInput] = None
    data_raw: Optional[str] = None

class ResultRequest(BaseModel):
    request_id: int = Field(..., gt=0)

class ApiStatus(BaseModel):
    status: str
    result: Optional[str] = None  # JSON as string to remain compatible with WP consumer
    message: Optional[str] = None


# -----------------------------------------------------------------------------
# Background processor
# -----------------------------------------------------------------------------
def process_job(payload: StartPayload):
    #try:
    data_for_llm = payload.data if payload.data is not None else payload.data_raw or {}
    raw = crisis_management_update2(data_for_llm)
    normalized = normalize_result_to_json_string(raw)
    save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=normalized)
    '''except Exception as e:
        # Store an error envelope so callers don’t spin forever
        err_json = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        save_result(request_id=payload.request_id, user_id=payload.user_id, result_text=err_json)'''


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start", response_model=ApiStatus)
def start(payload: StartPayload, bg: BackgroundTasks):
    """
    Asynchronous mode: enqueue background generation and return 'processing' immediately.
    WordPress can poll /result using request_id.
    """
    existing = fetch_latest_result(payload.request_id)
    if existing:
        return ApiStatus(status="done", result=existing["edited_result"] or existing["result"])

    bg.add_task(process_job, payload)
    return ApiStatus(status="processing")

@app.post("/start_sync", response_model=ApiStatus)
def start_sync(payload: StartPayload):
    """
    Synchronous mode: generate immediately and return 'done' + result.
    Useful for direct Postman tests or when you want the response inline.
    """
    existing = fetch_latest_result(payload.request_id)
    if existing:
        return ApiStatus(status="done", result=existing["edited_result"] or existing["result"])

    try:
        data_for_llm = payload.data if payload.data is not None else payload.data_raw or {}
        raw = crisis_management_update2(data_for_llm)
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
