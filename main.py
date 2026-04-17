from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from passlib.context import CryptContext
from dotenv import load_dotenv
from datetime import datetime
import requests, os, math, json, logging, io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import cloudinary
import cloudinary.uploader

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ecg")

# ── Cloudinary ────────────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key   =os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

# ── Database ──────────────────────────────────────────────────────────────────
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "")
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL missing in .env")

engine       = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

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
    id          = Column(Integer,  primary_key=True, index=True)
    user_id     = Column(Integer,  ForeignKey("users.id"), nullable=False)
    record_id   = Column(String,   nullable=True)
    filename    = Column(String,   nullable=True)
    image_url   = Column(String,   nullable=True)
    leads_count = Column(Integer,  nullable=True)
    snr         = Column(Float,    nullable=True)
    dice        = Column(Float,    nullable=True)
    duration    = Column(Float,    nullable=True)
    lead_list   = Column(Text,     nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    user        = relationship("UserDB", back_populates="scans")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:    yield db
    finally: db.close()

# ── Security ──────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def get_password_hash(p):           return pwd_context.hash(p)

# ── Config ────────────────────────────────────────────────────────────────────
HF_SPACE_URL             = os.getenv("HF_SPACE_URL", "https://bhemanth-ecg-hf-space.hf.space")
HF_API_ENDPOINT          = f"{HF_SPACE_URL.rstrip('/')}/analyze-ecg"
HF_TOKEN                 = os.getenv("HF_TOKEN", "")
GOOGLE_CLIENT_ID         = os.getenv("GOOGLE_CLIENT_ID", "")
WEBHOOK_VERIFY_TOKEN     = os.getenv("WEBHOOK_VERIFY_TOKEN", "ecg_digitizer_webhook_2024")
META_API_TOKEN           = os.getenv("META_API_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
META_API_VERSION         = "v20.0"

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_snr(signals: dict) -> float:
    snrs = []
    for lead, samples in signals.items():
        clean = [s for s in samples if s is not None and not math.isnan(s)]
        if len(clean) < 10: continue
        arr = np.array(clean, dtype=float)
        sp  = float(np.sum(arr**2))
        if sp == 0: continue
        snrs.append(round(10*math.log10(sp/(float(np.sum((arr-np.mean(arr))**2))+1e-9)), 2))
    return round(sum(snrs)/len(snrs), 2) if snrs else 0.0

def compute_duration(signals: dict) -> float:
    ii = signals.get("II") or signals.get("II_short") or []
    clean = [s for s in ii if s is not None and not math.isnan(s)]
    return round(len(clean)/500, 1) if clean else 0.0

# ── Schemas ───────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    full_name: str; email: str; password: str

class UserLogin(BaseModel):
    email: str; password: str

class GoogleLoginRequest(BaseModel):
    token: str

class UserProfileUpdate(BaseModel):
    email: str; full_name: str; age: int; gender: str; phone: str

class ChangePasswordRequest(BaseModel):
    email: str; new_password: str

class DeleteAccountRequest(BaseModel):
    email: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ECG Digitizer API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# =============================================================================
# PDF GENERATION
# =============================================================================

def _is_flat(sig, thr=0.05):
    if sig is None or len(sig) == 0: return True
    return float(np.ptp(np.asarray(sig, dtype=np.float32))) < thr


def generate_ecg_pdf(signals: dict, record_id: str = "",
                     lead_list: list = None, duration: float = None) -> bytes:
    """
    Render a medical-grade A4-landscape 12-lead ECG PDF and return raw bytes.

    Layout  :  3 rows × 4 leads (2.5 s each) + 1 rhythm row (10 s, Lead II)
    Grid    :  Minor 1 mm (0.04 s / 0.1 mV), Major 5 mm (0.2 s / 0.5 mV)
    Scale   :  25 mm/s  ×  10 mm/mV
    Calib   :  1 mV square pulse at left edge of every row
    """
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.4})

    DPI      = 200
    FIG_W    = 297 / 25.4   # A4 landscape width  (inches)
    FIG_H    = 210 / 25.4   # A4 landscape height

    PAPER_BG  = '#FFF5F5'
    MINOR_CLR = '#FFBBBB'
    MAJOR_CLR = '#FF6666'
    TRACE_CLR = '#111111'
    LW_MINOR  = 0.30
    LW_MAJOR  = 0.60
    LW_TRACE  = 0.50
    MINOR_S   = 0.04;  MAJOR_S  = 0.20
    MINOR_MV  = 0.1;   MAJOR_MV = 0.5

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=PAPER_BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    HEADER_H = 0.10
    FOOTER_H = 0.03
    BODY_H   = 1 - HEADER_H - FOOTER_H
    LEFT_F   = 0.005
    BODY_W   = 1 - 2*LEFT_F
    N_ROWS   = 4
    ROW_H_F  = BODY_H / N_ROWS

    # ── Compute stats ─────────────────────────────────────────────────────────
    all_leads = lead_list or list(signals.keys())
    n_valid   = sum(1 for l in all_leads if l != 'II_short' and not _is_flat(signals.get(l)))
    dur_str   = f"{duration:.1f}s" if duration else "—"
    now_str   = datetime.now().strftime("%d %b %Y  %H:%M")

    # ── Header ────────────────────────────────────────────────────────────────
    ax_h = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    ax_h.set_facecolor('#FFFFFF'); ax_h.axis('off')
    ax_h.axhline(0, color='#AAAAAA', lw=0.6, xmin=0.01, xmax=0.99)
    ax_h.text(0.012, 0.78, f'12-Lead ECG  ·  Record: {record_id or "—"}',
              transform=ax_h.transAxes, fontsize=9.5, fontweight='bold',
              color='#1a1a1a', va='top')
    ax_h.text(0.012, 0.30, f'{n_valid}/12 leads  ·  Duration: {dur_str}  ·  25 mm/s  ·  10 mm/mV',
              transform=ax_h.transAxes, fontsize=7, color='#555', va='top')
    ax_h.text(0.988, 0.78, now_str,
              transform=ax_h.transAxes, fontsize=7, color='#777', va='top', ha='right')
    ax_h.text(0.988, 0.30, 'Generated by ECG Digitizer AI',
              transform=ax_h.transAxes, fontsize=6.5, color='#AAA', va='top', ha='right')

    # ── Row definitions ───────────────────────────────────────────────────────
    ROW_DEFS = [
        (['I',        'aVR', 'V1', 'V4'],  ['I',   'aVR', 'V1', 'V4'],  2.5, 2.5),
        (['II_short', 'aVL', 'V2', 'V5'],  ['II',  'aVL', 'V2', 'V5'],  2.5, 2.5),
        (['III',      'aVF', 'V3', 'V6'],  ['III', 'aVF', 'V3', 'V6'],  2.5, 2.5),
        (['II'],                            ['II'],                       10.0, 2.0),
    ]

    def draw_row(ax, dur_s, row_leads, row_labels, ylim):
        ax.set_facecolor(PAPER_BG)
        ax.set_xlim(0, dur_s); ax.set_ylim(-ylim, ylim)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Grid
        for x in np.arange(0, dur_s+MINOR_S, MINOR_S):
            ax.axvline(x, color=MINOR_CLR, lw=LW_MINOR, zorder=1)
        for y in np.arange(-ylim, ylim+MINOR_MV, MINOR_MV):
            ax.axhline(y, color=MINOR_CLR, lw=LW_MINOR, zorder=1)
        for x in np.arange(0, dur_s+MAJOR_S, MAJOR_S):
            ax.axvline(x, color=MAJOR_CLR, lw=LW_MAJOR, zorder=2)
        for y in np.arange(-ylim, ylim+MAJOR_MV, MAJOR_MV):
            ax.axhline(y, color=MAJOR_CLR, lw=LW_MAJOR, zorder=2)

        n_seg   = len(row_leads)
        seg_dur = dur_s / n_seg if n_seg > 1 else dur_s

        for ci, (lname, ldisp) in enumerate(zip(row_leads, row_labels)):
            x0 = ci * seg_dur
            x1 = x0 + seg_dur

            # Calibration pulse (1 mV, 0.2 s)
            if ci == 0:
                ax.plot([x0, x0, x0+0.2, x0+0.2], [0, 1.0, 1.0, 0],
                        color=TRACE_CLR, lw=LW_TRACE*1.2, zorder=6)

            sig = signals.get(lname)
            if _is_flat(sig):
                ax.plot([x0, x1], [0, 0], color='#CC9999', lw=LW_TRACE,
                        ls='--', zorder=5, alpha=0.6)
                ax.text((x0+x1)/2, 0, 'No Signal', ha='center', va='center',
                        fontsize=4, color='#AA6666', fontstyle='italic', zorder=10)
            else:
                t = np.linspace(x0, x1, len(sig))
                ax.plot(t, sig, color=TRACE_CLR, lw=LW_TRACE, zorder=5,
                        solid_capstyle='round', solid_joinstyle='round')

            ax.text(x0 + dur_s*0.008, ylim*0.82, ldisp,
                    fontsize=5.5, fontweight='bold', color='#111',
                    zorder=10, va='top',
                    bbox=dict(facecolor=PAPER_BG, edgecolor='none', alpha=0.9, pad=0.4))
            if ci > 0:
                ax.axvline(x0, color='#888', lw=0.5, zorder=7)

    for ri, (r_leads, r_labels, dur, ylim) in enumerate(ROW_DEFS):
        yb = FOOTER_H + (N_ROWS-1-ri) * ROW_H_F
        ax = fig.add_axes([LEFT_F, yb, BODY_W, ROW_H_F])
        draw_row(ax, dur, r_leads, r_labels, ylim)
        if ri == N_ROWS-1:
            ax.text(dur-0.04, -ylim+0.08, '25 mm/s  ·  10 mm/mV',
                    ha='right', va='bottom', fontsize=4.5, color='#777', zorder=10)

    # Footer
    ax_ft = fig.add_axes([0, 0, 1, FOOTER_H])
    ax_ft.set_facecolor('#FFFFFF'); ax_ft.axis('off')
    ax_ft.axhline(1.0, color='#CCC', lw=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', dpi=DPI,
                bbox_inches='tight', facecolor=PAPER_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def upload_pdf_to_cloudinary(pdf_bytes: bytes, record_id: str) -> str | None:
    """Upload PDF bytes to Cloudinary and return the public secure_url."""
    try:
        fname  = f"ecg_report_{record_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        result = cloudinary.uploader.upload(
            pdf_bytes,
            folder        = "ecg_reports",
            public_id     = fname,
            resource_type = "raw",     # PDFs must be 'raw' in Cloudinary
            format        = "pdf",
        )
        url = result.get("secure_url")
        logger.info("PDF uploaded to Cloudinary: %s", url)
        return url
    except Exception as e:
        logger.error("Cloudinary PDF upload failed: %s", e)
        return None


# =============================================================================
# AUTH ENDPOINTS
# =============================================================================

@app.post("/api/signup")
def signup(body: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    db.add(UserDB(full_name=body.full_name, email=body.email,
                  hashed_password=get_password_hash(body.password)))
    db.commit()
    return {"message": "Account created successfully"}

@app.post("/api/login")
def login(body: UserLogin, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")
    return {"token": f"fake-jwt-{user.id}", "user": {
        "id": user.id, "name": user.full_name or "", "email": user.email,
        "age": user.age, "gender": user.gender, "phone": user.phone,
        "is_profile_complete": bool(user.age and user.gender and user.phone),
    }}

@app.post("/api/google-login")
def google_login(body: GoogleLoginRequest, db: Session = Depends(get_db)):
    try:
        idinfo = id_token.verify_oauth2_token(body.token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email  = idinfo["email"]; name = idinfo.get("name", "")
        user   = db.query(UserDB).filter(UserDB.email == email).first()
        if not user:
            user = UserDB(full_name=name, email=email,
                          hashed_password=get_password_hash(f"gso_{datetime.utcnow().timestamp()}"))
            db.add(user); db.commit(); db.refresh(user)
        return {"token": f"fake-jwt-{user.id}", "user": {
            "id": user.id, "name": user.full_name or "", "email": user.email,
            "age": user.age, "gender": user.gender, "phone": user.phone,
            "is_profile_complete": bool(user.age and user.gender and user.phone),
        }}
    except ValueError:
        raise HTTPException(401, "Invalid Google token")

@app.post("/api/update-profile")
def update_profile(body: UserProfileUpdate, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user: raise HTTPException(404, "User not found")
    user.full_name = body.full_name; user.age = body.age
    user.gender = body.gender; user.phone = body.phone
    db.commit()
    return {"message": "Profile updated"}

@app.post("/api/change-password")
def change_password(body: ChangePasswordRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user: raise HTTPException(404, "User not found")
    user.hashed_password = get_password_hash(body.new_password); db.commit()
    return {"message": "Password updated"}

@app.delete("/api/delete-account")
def delete_account(body: DeleteAccountRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == body.email).first()
    if not user: raise HTTPException(404, "User not found")
    db.delete(user); db.commit()
    return {"message": "Account deleted"}

# =============================================================================
# ECG UPLOAD  (web / app)
# =============================================================================

@app.post("/api/upload-ecg")
async def upload_ecg(file: UploadFile = File(...), email: str = None,
                     db: Session = Depends(get_db)):
    logger.info("FILE: %s", file.filename)
    content = await file.read()

    try:
        upload_result = cloudinary.uploader.upload(content, folder="ecg_uploads", resource_type="auto")
        image_url     = upload_result.get("secure_url")
    except Exception as e:
        raise HTTPException(500, f"Cloudinary upload failed: {e}")

    try:
        headers     = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        hf_response = requests.post(
            HF_API_ENDPOINT,
            files={"file": (file.filename, content, file.content_type or "image/png")},
            headers=headers, verify=False, timeout=180,
        )
        if hf_response.status_code == 503:
            raise HTTPException(503, "HF Space waking up — retry in ~30s")
        if hf_response.status_code != 200:
            raise HTTPException(hf_response.status_code,
                                f"HF Space error: {hf_response.text[:400]}")
        data = hf_response.json()
    except HTTPException: raise
    except requests.exceptions.Timeout:
        raise HTTPException(504, "HF Space timed out — retry in ~30s")
    except Exception as e:
        raise HTTPException(500, f"HF call failed: {e}")

    ecg_data  = data.get("data", data)
    signals   = ecg_data.get("signals",   {})
    lead_list = ecg_data.get("lead_list", [])
    snr       = compute_snr(signals)
    duration  = compute_duration(signals)
    record_id = file.filename.replace(".", "_").replace(" ", "_")

    if email:
        user = db.query(UserDB).filter(UserDB.email == email).first()
        if user:
            db.add(ScanDB(user_id=user.id, record_id=record_id, filename=file.filename,
                          image_url=image_url, leads_count=len(lead_list),
                          snr=snr, duration=duration, lead_list=json.dumps(lead_list),
                          created_at=datetime.utcnow()))
            db.commit()

    return {"status": "success", "data": {
        **ecg_data, "image_url": image_url,
        "computed_snr": snr, "computed_duration": duration,
    }}


# =============================================================================
# WHATSAPP HELPERS
# =============================================================================

def _wa_headers() -> dict:
    if not META_API_TOKEN:
        logger.error("META_API_TOKEN not set — WhatsApp replies will fail")
    return {"Authorization": f"Bearer {META_API_TOKEN}", "Content-Type": "application/json"}


def send_whatsapp_text(to: str, text: str, phone_number_id: str = None) -> bool:
    pid = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
    if not pid:
        logger.error("No WhatsApp phone_number_id — cannot send reply"); return False
    try:
        r = requests.post(
            f"https://graph.facebook.com/{META_API_VERSION}/{pid}/messages",
            headers=_wa_headers(),
            json={"messaging_product":"whatsapp","to":to,"type":"text","text":{"body":text}},
            timeout=15,
        )
        if r.status_code != 200:
            logger.error("WA text send failed (%d): %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        logger.error("WA text send exception: %s", e); return False


def send_whatsapp_document(to: str, pdf_url: str, filename: str,
                           caption: str, phone_number_id: str = None) -> bool:
    """
    Send a PDF document via WhatsApp using a public URL (Cloudinary).
    The URL must be HTTPS and publicly accessible without auth.
    """
    pid = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
    if not pid:
        logger.error("No WhatsApp phone_number_id — cannot send document"); return False
    try:
        r = requests.post(
            f"https://graph.facebook.com/{META_API_VERSION}/{pid}/messages",
            headers=_wa_headers(),
            json={
                "messaging_product": "whatsapp",
                "to":                to,
                "type":              "document",
                "document": {
                    "link":     pdf_url,
                    "filename": filename,
                    "caption":  caption,
                },
            },
            timeout=20,
        )
        if r.status_code != 200:
            logger.error("WA document send failed (%d): %s", r.status_code, r.text[:300])
            return False
        logger.info("WA PDF sent to %s: %s", to, filename)
        return True
    except Exception as e:
        logger.error("WA document send exception: %s", e); return False


def _download_meta_image(image_id: str) -> bytes | None:
    """Two-step Meta image download (resolve URL → download bytes)."""
    try:
        r = requests.get(
            f"https://graph.facebook.com/{META_API_VERSION}/{image_id}",
            headers=_wa_headers(), timeout=15,
        )
        if r.status_code != 200:
            logger.error("Media URL fetch failed (%d)", r.status_code); return None
        cdn_url = r.json().get("url")
        if not cdn_url:
            logger.error("No 'url' in media response"); return None
    except Exception as e:
        logger.error("Media URL fetch error: %s", e); return None

    try:
        r2 = requests.get(cdn_url,
                          headers={"Authorization": f"Bearer {META_API_TOKEN}"},
                          timeout=30)
        if r2.status_code != 200:
            logger.error("Image download failed (%d)", r2.status_code); return None
        return r2.content
    except Exception as e:
        logger.error("Image download error: %s", e); return None


def _call_hf_space(image_bytes: bytes, filename: str) -> dict | None:
    """Call HuggingFace ECG pipeline, return parsed dict or None."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        r = requests.post(
            HF_API_ENDPOINT,
            files={"file": (filename, image_bytes, "image/jpeg")},
            headers=headers, verify=False, timeout=180,
        )
        logger.info("HF status: %d", r.status_code)
        if r.status_code != 200:
            logger.error("HF error (%d): %s", r.status_code, r.text[:300]); return None
        return r.json()
    except requests.exceptions.Timeout:
        logger.error("HF timed out"); return None
    except Exception as e:
        logger.error("HF call error: %s", e); return None


# =============================================================================
# WHATSAPP BACKGROUND TASK
# =============================================================================

def process_whatsapp_image(image_id: str, sender_phone: str, phone_number_id: str):
    """
    Full pipeline for a WhatsApp ECG image:
      1. Download image from Meta CDN
      2. Call HuggingFace ECG AI
      3. Generate medical-grade PDF
      4. Upload PDF to Cloudinary (public HTTPS URL)
      5. Send PDF document + text summary back via WhatsApp
    """
    pid = phone_number_id or WHATSAPP_PHONE_NUMBER_ID

    # ── 1. Download image ────────────────────────────────────────────────────
    logger.info("[WA] Downloading image %s", image_id)
    image_bytes = _download_meta_image(image_id)
    if not image_bytes:
        send_whatsapp_text(sender_phone,
            "⚠️ Couldn't download your image from WhatsApp. Please send it again.", pid)
        return
    logger.info("[WA] Image: %d bytes", len(image_bytes))

    # ── 2. ECG AI analysis ───────────────────────────────────────────────────
    data = _call_hf_space(image_bytes, f"wa_{image_id}.jpg")
    if data is None:
        send_whatsapp_text(sender_phone,
            "⚠️ The ECG AI is waking up (cold start). Please wait 30 seconds and resend your image.", pid)
        return

    ecg_data  = data.get("data", data)
    if data.get("status") == "error" or "detail" in data:
        err = data.get("detail", "Unknown error")
        send_whatsapp_text(sender_phone, f"⚠️ Analysis failed: {str(err).capitalize()}", pid)
        return

    signals   = ecg_data.get("signals",   {})
    lead_list = ecg_data.get("lead_list", [])
    duration  = compute_duration(signals)
    n_valid   = ecg_data.get("valid_leads",
                    sum(1 for l in lead_list if l != "II_short" and not _is_flat(signals.get(l))))
    missing   = [l for l in ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
                 if l not in lead_list or _is_flat(signals.get(l))]

    record_id = f"WA_{image_id[:8].upper()}"
    logger.info("[WA] Analysis done: %d/12 valid leads", n_valid)

    # ── 3. Generate PDF ───────────────────────────────────────────────────────
    logger.info("[WA] Generating PDF...")
    try:
        pdf_bytes = generate_ecg_pdf(
            signals   = signals,
            record_id = record_id,
            lead_list = lead_list,
            duration  = duration,
        )
        logger.info("[WA] PDF: %d bytes", len(pdf_bytes))
    except Exception as e:
        logger.error("[WA] PDF generation failed: %s", e)
        # Fall back to text-only reply
        send_whatsapp_text(sender_phone,
            f"✅ ECG analysed: {n_valid}/12 leads, {duration}s — PDF generation failed, "
            f"view full report on the dashboard.", pid)
        return

    # ── 4. Upload PDF to Cloudinary ───────────────────────────────────────────
    logger.info("[WA] Uploading PDF to Cloudinary...")
    pdf_url = upload_pdf_to_cloudinary(pdf_bytes, record_id)

    if not pdf_url:
        # Cloudinary upload failed — send text summary instead
        logger.warning("[WA] Falling back to text-only reply")
        summary = (
            f"✅ *ECG Analysis Complete!*\n\n"
            f"📊 Leads: {n_valid}/12\n"
            f"⏱ Duration: {duration}s\n"
        )
        if missing:
            summary += f"⚠️ Not detected: {', '.join(missing)}\n"
        summary += "\n📱 Full PDF report: https://ecg-digitizer.vercel.app"
        send_whatsapp_text(sender_phone, summary, pid)
        return

    # ── 5. Send PDF document via WhatsApp ─────────────────────────────────────
    pdf_filename = f"ECG_Report_{record_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    caption_lines = [
        f"✅ *12-Lead ECG Report*",
        f"📊 {n_valid}/12 leads  ·  ⏱ {duration}s",
    ]
    if missing:
        caption_lines.append(f"⚠️ Not detected: {', '.join(missing)}")
    caption_lines += [
        "",
        "Open the PDF for the full medical grid with calibrated amplitudes.",
        "🌐 Dashboard: https://ecg-digitizer.vercel.app",
    ]
    caption = "\n".join(caption_lines)

    ok = send_whatsapp_document(sender_phone, pdf_url, pdf_filename, caption, pid)

    if not ok:
        # Document send failed (e.g. URL not yet public) — send link as text
        send_whatsapp_text(sender_phone,
            f"✅ ECG Report ready!\n📊 {n_valid}/12 leads  ·  ⏱ {duration}s\n\n"
            f"📄 Download PDF: {pdf_url}", pid)

    logger.info("[WA] Completed for %s", sender_phone)


# =============================================================================
# WHATSAPP WEBHOOK ENDPOINTS
# =============================================================================

@app.get("/whatsapp-webhook")
async def verify_webhook(request: Request):
    """
    Meta webhook verification.
    Set in Meta Dashboard → WhatsApp → Configuration:
      Callback URL  : https://<your-railway-url>/whatsapp-webhook
      Verify token  : ecg_digitizer_webhook_2024
    Then subscribe to the 'messages' field.
    """
    mode      = request.query_params.get("hub.mode")
    token     = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    logger.info("Webhook verify — mode=%s match=%s", mode, token == WEBHOOK_VERIFY_TOKEN)
    if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Forbidden", status_code=403)


@app.post("/whatsapp-webhook")
async def receive_whatsapp_message(request: Request, background_tasks: BackgroundTasks):
    """Receive all incoming WhatsApp events. Must return 200 immediately."""
    try:
        body = await request.json()
    except Exception:
        return {"status": "ok"}

    try:
        entry = body.get("entry", [])
        if not entry: return {"status": "ok"}
        value = entry[0].get("changes", [{}])[0].get("value", {})

        # Ignore delivery/read status updates
        if "statuses" in value and "messages" not in value:
            return {"status": "ok"}

        messages = value.get("messages", [])
        if not messages: return {"status": "ok"}

        message         = messages[0]
        sender_phone    = message.get("from", "")
        phone_number_id = value.get("metadata", {}).get("phone_number_id", WHATSAPP_PHONE_NUMBER_ID)
        msg_type        = message.get("type", "")

        logger.info("[WA] %s from %s", msg_type, sender_phone)

        if msg_type == "image":
            image_id = message["image"]["id"]
            # Immediate acknowledgement
            send_whatsapp_text(
                sender_phone,
                "🫀 ECG image received!\n\nAnalysing your 12-lead ECG — you'll receive the PDF report in about 30 seconds...",
                phone_number_id,
            )
            # Heavy work in background
            background_tasks.add_task(
                process_whatsapp_image, image_id, sender_phone, phone_number_id
            )

        elif msg_type == "text":
            txt = message.get("text", {}).get("body", "").strip().lower()
            if any(k in txt for k in ["hi","hello","start","help","ecg"]):
                send_whatsapp_text(sender_phone,
                    "👋 Hello! I'm the *ECG Digitizer Bot*.\n\n"
                    "📸 Send me a clear photo of a *12-lead ECG printout* and I'll:\n"
                    "   • Extract all 12 leads automatically\n"
                    "   • Measure every signal in mV\n"
                    "   • Send you back a *medical-grade PDF report* 📄\n\n"
                    "Just send the image to get started! 🫀",
                    phone_number_id,
                )
            else:
                send_whatsapp_text(sender_phone,
                    "📸 Please send a clear photo of your 12-lead ECG to get started.",
                    phone_number_id,
                )
        else:
            send_whatsapp_text(sender_phone,
                "📸 I only process ECG images. Please send a photo of your 12-lead ECG printout.",
                phone_number_id,
            )

    except Exception as e:
        logger.exception("[WA] Webhook error: %s", e)

    return {"status": "ok"}


# =============================================================================
# SCAN HISTORY
# =============================================================================

@app.get("/api/scan-history")
def scan_history(email: str, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user: raise HTTPException(404, "User not found")
    scans = db.query(ScanDB).filter(ScanDB.user_id == user.id).order_by(ScanDB.created_at.desc()).all()
    rows  = [{
        "id":          s.id,
        "record_id":   s.record_id   or "—",
        "filename":    s.filename    or "—",
        "image_url":   s.image_url,
        "leads_count": s.leads_count or 0,
        "snr":         f"{s.snr:.1f} dB" if s.snr is not None else "—",
        "duration":    f"{s.duration:.1f}s" if s.duration is not None else "—",
        "lead_list":   json.loads(s.lead_list) if s.lead_list else [],
        "created_at":  s.created_at.strftime("%d %b %Y, %I:%M %p") if s.created_at else "—",
    } for s in scans]
    dv = [s.duration    for s in scans if s.duration    is not None]
    lv = [s.leads_count for s in scans if s.leads_count is not None]
    return {
        "total_scans":  len(scans),
        "avg_leads":    round(sum(lv)/len(lv)) if lv else "—",
        "avg_duration": f"{sum(dv)/len(dv):.1f}s" if dv else "—",
        "scans":        rows,
    }

@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    s = db.query(ScanDB).filter(ScanDB.id == scan_id).first()
    if not s: raise HTTPException(404, "Scan not found")
    return {
        "id": s.id, "record_id": s.record_id, "filename": s.filename,
        "image_url": s.image_url, "leads_count": s.leads_count,
        "snr": s.snr, "duration": s.duration,
        "lead_list": json.loads(s.lead_list) if s.lead_list else [],
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }

# =============================================================================
# HEALTH
# =============================================================================

@app.get("/")
def root():
    return {
        "status":   "ECG Digitizer API v2",
        "whatsapp": "ready" if META_API_TOKEN else "NOT CONFIGURED — set META_API_TOKEN",
    }

@app.get("/health")
def health():
    return {
        "status":                "ok",
        "timestamp":             datetime.utcnow().isoformat(),
        "hf_endpoint":           HF_API_ENDPOINT,
        "whatsapp_token_set":    bool(META_API_TOKEN),
        "whatsapp_phone_id_set": bool(WHATSAPP_PHONE_NUMBER_ID),
        "webhook_verify_token":  WEBHOOK_VERIFY_TOKEN,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)