from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from passlib.context import CryptContext
from dotenv import load_dotenv
from datetime import datetime
import requests, os, math, json

# --- Google Imports ---
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

import cloudinary
import cloudinary.uploader

load_dotenv()

# ── Cloudinary config ─────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

# ── Database ──────────────────────────────────────────────────────────────────
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL missing in .env")

engine       = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

# ── Models ────────────────────────────────────────────────────────────────────
class UserDB(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    full_name       = Column(String,  nullable=True)
    email           = Column(String,  unique=True, index=True, nullable=False)
    hashed_password = Column(String,  nullable=False)
    age             = Column(Integer, nullable=True)
    gender          = Column(String,  nullable=True)
    phone           = Column(String,  nullable=True)
    scans           = relationship("ScanDB", back_populates="user", cascade="all, delete-orphan")

class ScanDB(Base):
    __tablename__ = "scans"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    record_id   = Column(String,  nullable=True)
    filename    = Column(String,  nullable=True)
    image_url   = Column(String,  nullable=True)
    leads_count = Column(Integer, nullable=True)
    snr         = Column(Float,   nullable=True)
    dice        = Column(Float,   nullable=True)
    duration    = Column(Float,   nullable=True)
    lead_list   = Column(Text,    nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    user        = relationship("UserDB", back_populates="scans")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Security ──────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def get_password_hash(password):    return pwd_context.hash(password)

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_snr(signals: dict) -> float:
    import numpy as np
    snrs = []
    for lead, samples in signals.items():
        clean = [s for s in samples if s is not None and not math.isnan(s)]
        if len(clean) < 10:
            continue
        arr = np.array(clean, dtype=float)
        sig_power   = float(np.sum(arr ** 2))
        if sig_power == 0:
            continue
        noise       = arr - np.mean(arr)
        noise_power = float(np.sum(noise ** 2)) + 1e-9
        snr         = 10 * math.log10(sig_power / noise_power)
        snrs.append(round(snr, 2))
    return round(sum(snrs) / len(snrs), 2) if snrs else 0.0

def compute_duration(signals: dict) -> float:
    ii    = signals.get("II") or signals.get("II_short") or []
    clean = [s for s in ii if s is not None and not math.isnan(s)]
    return round(len(clean) / 500, 1) if clean else 0.0

# ── Schemas ───────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    full_name: str
    email:     str
    password:  str

class UserLogin(BaseModel):
    email:    str
    password: str

class GoogleLoginRequest(BaseModel):
    token: str

class UserProfileUpdate(BaseModel):
    email:     str
    full_name: str
    age:       int
    gender:    str
    phone:     str

class ChangePasswordRequest(BaseModel):
    email:        str
    new_password: str

class DeleteAccountRequest(BaseModel):
    email: str

# ── App & Config ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_SPACE_URL        = os.getenv("HF_SPACE_URL", "https://bhemanth-ecg-hf-space.hf.space")
HF_API_ENDPOINT     = f"{HF_SPACE_URL.rstrip('/')}/analyze-ecg"
HF_TOKEN            = os.getenv("HF_TOKEN", None)
GOOGLE_CLIENT_ID    = os.getenv("GOOGLE_CLIENT_ID", "")

# ── WHATSAPP WEBHOOK CONFIG ──
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "my_super_secret_ecg_token_123")
META_API_TOKEN       = os.getenv("META_API_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/signup")
def signup(body: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = UserDB(
        full_name       = body.full_name,
        email           = body.email,
        hashed_password = get_password_hash(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Account created successfully"}


@app.post("/api/login")
def login(body: UserLogin, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    is_profile_complete = bool(user.age and user.gender and user.phone)

    return {
        "token": f"fake-jwt-{user.id}",
        "user": {
            "id":                  user.id,
            "name":                user.full_name or "",
            "email":               user.email,
            "age":                 user.age,
            "gender":              user.gender,
            "phone":               user.phone,
            "is_profile_complete": is_profile_complete,
        }
    }

@app.post("/api/google-login")
def google_login(body: GoogleLoginRequest, db: Session = Depends(get_db)):
    try:
        # Verify the token via Google's library
        idinfo = id_token.verify_oauth2_token(
            body.token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        email = idinfo['email']
        name = idinfo.get('name', '')
        
        # Check if user already exists
        user = db.query(UserDB).filter(UserDB.email == email).first()
        
        if not user:
            # Create a new user account if they don't exist
            # Generating a random/dummy hash since they will auth via Google
            dummy_password = f"google_sso_{datetime.utcnow().timestamp()}"
            user = UserDB(
                full_name       = name,
                email           = email,
                hashed_password = get_password_hash(dummy_password),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        is_profile_complete = bool(user.age and user.gender and user.phone)

        return {
            "token": f"fake-jwt-{user.id}",
            "user": {
                "id":                  user.id,
                "name":                user.full_name or "",
                "email":               user.email,
                "age":                 user.age,
                "gender":              user.gender,
                "phone":               user.phone,
                "is_profile_complete": is_profile_complete,
            }
        }
    except ValueError:
        # Invalid token
        raise HTTPException(status_code=401, detail="Invalid Google authentication token")


@app.post("/api/update-profile")
def update_profile(body: UserProfileUpdate, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.full_name = body.full_name
    user.age       = body.age
    user.gender    = body.gender
    user.phone     = body.phone
    db.commit()
    return {"message": "Profile updated"}


@app.post("/api/change-password")
def change_password(body: ChangePasswordRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    return {"message": "Password updated"}


@app.delete("/api/delete-account")
def delete_account(body: DeleteAccountRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": "Account deleted"}

# ─────────────────────────────────────────────────────────────────────────────
# ECG UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload-ecg")
async def upload_ecg(
    file:  UploadFile = File(...),
    email: str        = None,
    db:    Session    = Depends(get_db)
):
    print("STEP 1: FILE RECEIVED —", file.filename, file.content_type)

    content = await file.read()

    # ── Step 2: Upload to Cloudinary ─────────────────────────────────────────
    try:
        upload_result = cloudinary.uploader.upload(
            content,
            folder        = "ecg_uploads",
            resource_type = "auto"
        )
        image_url = upload_result.get("secure_url")
        print("STEP 2: CLOUDINARY SUCCESS — URL:", image_url)
    except Exception as e:
        print("CLOUDINARY ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {e}")

    # ── Step 3: Call HuggingFace Space ────────────────────────────────────────
    try:
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        print("STEP 3: CALLING HF SPACE →", HF_API_ENDPOINT)

        hf_response = requests.post(
            HF_API_ENDPOINT,
            files   = {"file": (file.filename, content, file.content_type or "image/png")},
            headers = headers,
            verify  = False,
            timeout = 180,
        )

        print("HF STATUS:", hf_response.status_code)

        if hf_response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail=(
                    "HuggingFace Space is waking up (cold start). "
                    "Please wait ~30 seconds and try again."
                )
            )

        if hf_response.status_code == 404:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"HF Space endpoint not found (404). "
                    f"URL tried: {HF_API_ENDPOINT}. "
                    f"Fix: Go to https://huggingface.co/spaces, open your Space, "
                    f"click the App tab, copy the exact URL, and set "
                    f"HF_SPACE_URL=<that_url> in your .env file."
                )
            )

        if hf_response.status_code != 200:
            raise HTTPException(
                status_code=hf_response.status_code,
                detail=(
                    f"HF Space returned HTTP {hf_response.status_code}. "
                    f"Response: {hf_response.text[:400]}"
                )
            )

        data = hf_response.json()
        print("STEP 4: HF RESPONSE PARSED OK — leads:", len(data.get("data", data).get("lead_list", [])))

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=(
                "HuggingFace Space request timed out after 180s. "
                "The Space may be in cold start mode — wait 30 seconds and retry."
            )
        )
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to HuggingFace Space at {HF_API_ENDPOINT}. Error: {e}"
        )
    except Exception as e:
        print("HF CALL ERROR:", e)
        raise HTTPException(status_code=500, detail=f"HF Space call failed: {e}")

    # ── Step 4: Parse, compute metrics, save to DB ────────────────────────────
    try:
        ecg_data  = data.get("data", data)
        signals   = ecg_data.get("signals",   {})
        lead_list = ecg_data.get("lead_list", [])

        snr       = compute_snr(signals)
        duration  = compute_duration(signals)
        record_id = file.filename.replace(".", "_").replace(" ", "_")

        if email:
            user = db.query(UserDB).filter(UserDB.email == email).first()
            if user:
                scan = ScanDB(
                    user_id     = user.id,
                    record_id   = record_id,
                    filename    = file.filename,
                    image_url   = image_url,
                    leads_count = len(lead_list),
                    snr         = snr,
                    duration    = duration,
                    lead_list   = json.dumps(lead_list),
                    created_at  = datetime.utcnow(),
                )
                db.add(scan)
                db.commit()
                print("STEP 5: SCAN SAVED TO DB — id:", scan.id)

        return {
            "status": "success",
            "data": {
                **ecg_data,
                "image_url":         image_url,
                "computed_snr":      snr,
                "computed_duration": duration,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print("DB/PARSE ERROR:", e)
        raise HTTPException(status_code=500, detail=f"Error saving scan result: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# WHATSAPP META WEBHOOK (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def send_whatsapp_text(phone_number_id: str, to_number: str, text: str):
    """Helper function to send a standard text reply via Meta API"""
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {META_API_TOKEN}", 
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text}
    }
    requests.post(url, headers=headers, json=data)

def process_meta_image_bg(image_id: str, sender_phone: str, phone_number_id: str):
    """Background task to download the image from Meta, run AI, and reply."""
    try:
        # 1. Ask Meta for the Image Download URL
        url_req = requests.get(
            f"https://graph.facebook.com/v18.0/{image_id}", 
            headers={"Authorization": f"Bearer {META_API_TOKEN}"}
        )
        image_url = url_req.json().get('url')
        
        if not image_url:
            send_whatsapp_text(phone_number_id, sender_phone, "⚠️ Could not download image from WhatsApp servers.")
            return

        # 2. Download the image bytes securely from Meta
        img_response = requests.get(
            image_url, 
            headers={"Authorization": f"Bearer {META_API_TOKEN}"}
        )
        content = img_response.content

        # 3. Forward the image to your existing Hugging Face Space ML logic
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        hf_response = requests.post(
            HF_API_ENDPOINT,
            files={"file": (f"wa_image_{image_id}.jpg", content, "image/jpeg")},
            headers=headers,
            verify=False,
            timeout=180,
        )

        # 4. Handle Hugging Face custom rejections / errors
        if hf_response.status_code != 200:
            try:
                error_detail = hf_response.json().get("detail", "Error connecting to AI.")
            except:
                error_detail = "Failed to process image."
            
            # Send the exact rejection text back via WhatsApp
            send_whatsapp_text(phone_number_id, sender_phone, f"⚠️ {str(error_detail).capitalize()}")
            return

        # 5. Successfully Processed! Format and send the response
        data = hf_response.json()
        ecg_data = data.get("data", data)
        lead_list = ecg_data.get("lead_list", [])
        duration = compute_duration(ecg_data.get("signals", {}))

        reply = (
            f"✅ *Analysis Complete!*\n"
            f"Extracted {len(lead_list)}/13 leads.\n"
            f"Duration: {duration}s\n\n"
            f"Log in to the web dashboard to view the full graphical report and download the PDF."
        )
        send_whatsapp_text(phone_number_id, sender_phone, reply)

    except Exception as e:
        print(f"Meta Background Task Error: {e}")
        send_whatsapp_text(phone_number_id, sender_phone, "⚠️ An error occurred while processing your ECG.")

@app.get("/whatsapp-webhook")
async def verify_webhook(request: Request):
    """Meta will send a GET request here once to verify your server endpoint."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
        # Meta requires you to return the raw challenge integer
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Forbidden", status_code=403)

@app.post("/whatsapp-webhook")
async def receive_whatsapp_message(request: Request, background_tasks: BackgroundTasks):
    """Meta will POST JSON data here every time someone sends your bot a message."""
    body = await request.json()
    try:
        if 'entry' in body and body['entry']:
            entry = body['entry'][0]['changes'][0]['value']
            
            if 'messages' in entry:
                message = entry['messages'][0]
                sender_phone = message['from']
                phone_number_id = entry['metadata']['phone_number_id']

                # Check if they sent an image
                if message['type'] == 'image':
                    image_id = message['image']['id']
                    
                    # Send an immediate text back acknowledging receipt
                    send_whatsapp_text(
                        phone_number_id, 
                        sender_phone, 
                        "🫀 Image received! Digitizing the 12-lead ECG. Please wait ~30 seconds..."
                    )
                    
                    # Run the heavy AI in the background so we return HTTP 200 immediately
                    background_tasks.add_task(process_meta_image_bg, image_id, sender_phone, phone_number_id)
                else:
                    # They sent a regular text or file type we don't support
                    send_whatsapp_text(
                        phone_number_id, 
                        sender_phone, 
                        "Hello! Please send a clear image of a 12-Lead ECG for me to analyze."
                    )
                    
        return {"status": "success"}
    except Exception as e:
        print(f"Webhook Error: {e}")
        return {"status": "error"}

# ─────────────────────────────────────────────────────────────────────────────
# SCAN HISTORY 
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/scan-history")
def scan_history(email: str, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    scans = (
        db.query(ScanDB)
        .filter(ScanDB.user_id == user.id)
        .order_by(ScanDB.created_at.desc())
        .all()
    )

    scan_list = []
    for s in scans:
        scan_list.append({
            "id":          s.id,
            "record_id":   s.record_id   or "—",
            "filename":    s.filename    or "—",
            "image_url":   s.image_url   or None,
            "leads_count": s.leads_count or 0,
            "snr":         f"{s.snr:.1f} dB" if s.snr is not None else "—",
            "duration":    f"{s.duration:.1f}s" if s.duration is not None else "—",
            "lead_list":   json.loads(s.lead_list) if s.lead_list else [],
            "created_at":  s.created_at.strftime("%d %b %Y, %I:%M %p") if s.created_at else "—",
        })

    total_scans  = len(scans)
    dur_values   = [s.duration for s in scans if s.duration is not None]
    lead_values  = [s.leads_count for s in scans if s.leads_count is not None]

    avg_duration = f"{sum(dur_values) / len(dur_values):.1f}s" if dur_values else "—"
    avg_leads    = round(sum(lead_values) / len(lead_values)) if lead_values else "—"

    return {
        "total_scans":  total_scans,
        "avg_leads":    avg_leads,
        "avg_duration": avg_duration,
        "scans":        scan_list,
    }

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SCAN DETAIL 
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(ScanDB).filter(ScanDB.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {
        "id":          scan.id,
        "record_id":   scan.record_id,
        "filename":    scan.filename,
        "image_url":   scan.image_url,
        "leads_count": scan.leads_count,
        "snr":         scan.snr,
        "duration":    scan.duration,
        "lead_list":   json.loads(scan.lead_list) if scan.lead_list else [],
        "created_at":  scan.created_at.isoformat() if scan.created_at else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ECG Digitizer API running"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)