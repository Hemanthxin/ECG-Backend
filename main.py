from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from passlib.context import CryptContext
from dotenv import load_dotenv
from datetime import datetime
import requests, os, math, json

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

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HuggingFace Space URL ─────────────────────────────────────────────────────
# HOW TO FIND YOUR CORRECT URL:
#   1. Go to https://huggingface.co/spaces
#   2. Find your Space → click it → click the "App" tab
#   3. The browser URL will look like: https://bhemanth-ecg-digitizer.hf.space
#   4. Set HF_SPACE_URL=https://bhemanth-<your-exact-space-name>.hf.space in your .env
#
# Common mistake: Space name in URL uses hyphens not underscores, e.g.
#   Space named "ecg hf space" → URL is "bhemanth-ecg-hf-space.hf.space"
#   Space named "ECG_Digitizer" → URL is "bhemanth-ecg-digitizer.hf.space"
#
HF_SPACE_URL      = os.getenv("HF_SPACE_URL", "https://bhemanth-ecg-hf-space.hf.space")
HF_API_ENDPOINT   = f"{HF_SPACE_URL.rstrip('/')}/analyze-ecg"
HF_TOKEN          = os.getenv("HF_TOKEN", None)   # only needed for private Spaces

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
            timeout = 180,      # HF Spaces need extra time on cold start
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
                status_code=500,
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
# SCAN HISTORY  ← THIS WAS THE MISSING ENDPOINT CAUSING 404
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/scan-history")
def scan_history(email: str, db: Session = Depends(get_db)):
    # Find user
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get all scans for this user, newest first
    scans = (
        db.query(ScanDB)
        .filter(ScanDB.user_id == user.id)
        .order_by(ScanDB.created_at.desc())
        .all()
    )

    # Build scan list
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

    # Aggregate stats
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
# INDIVIDUAL SCAN DETAIL (bonus — used by UploadECG history if needed)
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