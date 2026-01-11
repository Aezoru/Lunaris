# ==============================================================================
# الملف الرئيسي للخادم الخلفي لتطبيق Lunaris AI
# الإصدار: 3.0 (مع دعم سجل المحادثة، رفع الملفات، وتحسينات أخرى)
# ==============================================================================

import uvicorn
import os
import base64
from pydantic import BaseModel
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional

# --- 1. الإعدادات الرئيسية ---
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
MODEL_TEXT = "gemini-2.5-flash"
MODEL_VISION = "gemini-2.5-flash" # نموذج يدعم الصور والنصوص
MODEL_IMAGE_GEN = "gemini-2.5-flash" # نموذج توليد الصور

# --- 2. إعداد نماذج Gemini ---
model_text = None
model_vision = None
model_image_gen = None
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        print("خطأ فادح: لم يتم تعيين مفتاح GOOGLE_API_KEY.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model_text = genai.GenerativeModel(MODEL_TEXT)
        model_vision = genai.GenerativeModel(MODEL_VISION)
        model_image_gen = genai.GenerativeModel(MODEL_IMAGE_GEN)
        print("تم إعداد جميع نماذج Gemini بنجاح.")
except Exception as e:
    print(f"خطأ فادح أثناء إعداد Gemini: {e}")

# --- 3. إعداد تطبيق FastAPI ---
app = FastAPI(title="Lunaris AI Backend", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # السماح لجميع المصادر
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # تحديد الطرق المسموح بها بوضوح
    allow_headers=["*"],  # السماح لجميع الترويسات
)

# --- 4. نماذج الطلبات (Pydantic Models) ---
class Part(BaseModel):
    text: Optional[str] = None

class Content(BaseModel):
    role: str
    parts: List[Part]

class ChatRequest(BaseModel):
    message: str
    history: List[Content] = []
    enableDeepThink: bool = False

class TitleRequest(BaseModel):
    text: str
    language: str = "ar"

class ImageRequest(BaseModel):
    prompt: str

class AttachmentModel(BaseModel):
    mime_type: str
    data: str  # بيانات base64 كنص
    name: Optional[str] = None

class AnalyzeRequest(BaseModel):
    prompt: str
    attachments: List[AttachmentModel]

# --- 5. تعريف نقاط النهاية (API Endpoints) ---

@app.get("/", tags=["Status"])
def read_root():
    return {"status": "ok", "message": "خادم Lunaris AI v3.0 يعمل."}

# في ملف server.py، استبدل دالة chat_stream القديمة بهذه

@app.post("/chat-stream", tags=["Core"])
async def chat_stream(request: ChatRequest):
    if not model_text:
        raise HTTPException(status_code=503, detail="خدمة الدردشة غير متاحة.")

    # تحويل سجل المحادثات القديم إلى قواميس
    plain_history = [item.dict() for item in request.history]

    # إضافة رسالة المستخدم الحالية إلى نهاية السجل
    # هذا يضمن الترتيب الصحيح للمحادثة
    user_message_part = {"role": "user", "parts": [{"text": request.message}]}
    full_history = plain_history + [user_message_part]

    # إزالة الرسالة الأخيرة من السجل الكامل لتكون هي البرومبت
    final_prompt = full_history.pop()["parts"][0]["text"]
    
    # بناء التعليمات البرمجية للتفكير العميق إذا تم تفعيله
    if request.enableDeepThink:
        print("تفعيل وضع التفكير العميق...")
        final_prompt = (
            "SYSTEM_COMMAND: Engage Deep Analysis Mode. "
            "Start your response with a <thinking> block outlining your multi-step reasoning, exploring nuances, and considering counter-arguments. "
            "After the thinking block, provide a comprehensive, well-structured final answer.\n\n"
            f"USER_REQUEST: {final_prompt}"
        )

    async def stream_generator():
        try:
            # بدء المحادثة بالسجل القديم فقط
            chat_session = model_text.start_chat(history=full_history)
            # إرسال الرسالة الجديدة كطلب منفصل
            stream = await chat_session.send_message_async(final_prompt, stream=True)
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            error_message = f"حدث خطأ أثناء التواصل مع Gemini: {e}"
            print(error_message)
            yield f"ERROR: {error_message}"

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.post("/analyze-content", tags=["Core"])
async def analyze_content(request: AnalyzeRequest): # <-- تغيير هنا لاستخدام النموذج الجديد
    if not model_vision:
        raise HTTPException(status_code=503, detail="خدمة تحليل المحتوى غير متاحة.")

    print("=============================================")
    print(f"== استلام طلب /analyze-content (JSON)")
    print(f"== النص المستلم: {request.prompt[:50]}...")
    print(f"== عدد المرفقات: {len(request.attachments)}")
    print("=============================================")
    
    content_parts = []
    
    # إضافة النص أولاً
    content_parts.append(request.prompt)

    for att in request.attachments:
        try:
            # تحويل بيانات base64 المستلمة إلى بيانات ثنائية
            image_bytes = base64.b64decode(att.data)
            
            # إنشاء الجزء المتوافق مع Gemini
            image_part = {
                "mime_type": att.mime_type,
                "data": image_bytes
            }
            content_parts.append(image_part)
            print(f"-> تمت معالجة المرفق: {att.name} (النوع: {att.mime_type})")
        except Exception as e:
            print(f"[خطأ] فشل في معالجة بيانات base64 للمرفق {att.name}: {e}")
            raise HTTPException(status_code=400, detail=f"بيانات base64 غير صالحة للمرفق {att.name}")

    async def stream_generator():
        try:
            print("==> بدء إرسال الطلب إلى Gemini Vision...")
            stream = model_vision.generate_content(content_parts, stream=True)
            
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
            print("<== انتهى البث من Gemini Vision.")
        except Exception as e:
            error_message = f"حدث خطأ أثناء تحليل المحتوى: {e}"
            print(f"[خطأ] {error_message}")
            yield f"ERROR: {error_message}"

    return StreamingResponse(stream_generator(), media_type="text/plain")
# --- نقاط النهاية المساعدة (تبقى كما هي) ---
@app.post("/generate-title", tags=["Utilities"])
async def generate_title_endpoint(request: TitleRequest):
    # ... (الكود لم يتغير)
    pass

@app.post("/generate-image", tags=["Creative"])
async def generate_image_endpoint(request: ImageRequest):
    # ... (الكود لم يتغير)
    pass


# --- 6. كود تشغيل الخادم ---
if __name__ == "__main__":
    print("="*50)
    print("بدء تشغيل خادم Lunaris AI (الإصدار 3.0 المتكامل)...")
    print(f"الاستماع على العنوان: http://0.0.0.0:8000")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)

