import os
import uuid
import datetime
import base64
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# AI imports
import torch
from diffusers import StableDiffusionPipeline
from deep_translator import GoogleTranslator
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "static" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "change_this_secret"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR/'app.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- Models ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    kind = db.Column(db.String(32))  # generate | caption | speech
    image_filename = db.Column(db.String(300))
    caption = db.Column(db.Text)      # styled prompt or speech text
    translated = db.Column(db.Text)   # English translation
    original_input = db.Column(db.Text)  # original text input
    converted = db.Column(db.Text)    # converted prompt
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    user = db.relationship("User", backref=db.backref("histories", lazy=True))


with app.app_context():
    db.create_all()

# ---------------- Stable Diffusion ----------------
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    ).to("cpu")   # NSFW safety checker enabled by default
    SD_AVAILABLE = True
except Exception as e:
    SD_AVAILABLE = False
    print("Stable Diffusion load failed:", e)

# ---------------- Captioning Model ----------------
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    CAPTION_AVAILABLE = True
except Exception as e:
    CAPTION_AVAILABLE = False
    print("Caption model load failed:", e)

# ---------------- Helpers ----------------
def unique_name(orig):
    return f"{uuid.uuid4().hex[:8]}_{secure_filename(orig)}"

def convert_to_wav(input_path):
    """Convert any audio format to wav for SpeechRecognition"""
    output_path = str(Path(input_path).with_suffix(".wav"))
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print("Conversion failed:", e)
        return None

# ---------------- Content Safety ----------------
UNSAFE_KEYWORDS = [
    "nude", "porn", "sex", "nsfw", "blood", "gore", "kill", "murder",
    "violence", "abuse", "weapon", "gun", "drugs", "suicide", "terrorism"
]

def is_safe_text(text: str) -> bool:
    """Check if text contains unsafe words"""
    for word in UNSAFE_KEYWORDS:
        if word.lower() in text.lower():
            return False
    return True

def is_safe_image(image_path: str) -> bool:
    """
    Placeholder for advanced NSFW detection.
    Currently always returns True (Safe).
    You can integrate a pretrained NSFW detection model here.
    """
    return True

# ---------------- User Loader ----------------
@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# ---------------- Routes ----------------
@app.route("/")
def root():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if not username or not password:
            flash("Provide username and password", "danger")
            return render_template("register.html")
        if User.query.filter_by(username=username).first():
            flash("User exists", "danger")
            return render_template("register.html")
        u = User(username=username)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        flash("Account created, please log in", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        u = User.query.filter_by(username=username).first()
        if u and u.check_password(password):
            login_user(u)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

# ---------------- Translate Prompt API ----------------
@app.route("/translate_prompt", methods=["POST"])
@login_required
def translate_prompt():
    data = request.get_json()
    text = data.get("text", "")
    target = data.get("target", "")

    if not text or not target:
        return jsonify({"converted": "", "english": "", "engine": ""})

    try:
        # Convert into selected script
        converted = GoogleTranslator(source="auto", target=target).translate(text)

        # English meaning
        english = GoogleTranslator(source="auto", target="en").translate(text)

        if english.strip().lower() == text.strip().lower():
            english = GoogleTranslator(source="auto", target="en").translate(converted)

        engine = "deep-translator"
    except Exception as e:
        print("Translation error:", e)
        converted, english, engine = text, text, "error"

    return jsonify({"converted": converted, "english": english, "engine": engine})

# ---------------- Generate ----------------
@app.route("/generate", methods=["GET","POST"])
@login_required
def generate():
    if request.method == "POST":
        text_prompt = request.form.get("text_prompt", "").strip()
        converted = request.form.get("converted_prompt", "")
        prompt_en = request.form.get("english_prompt", "")
        style = request.form.get("style", "Natural")

        if not text_prompt:
            flash("No prompt provided", "danger")
            return redirect(url_for("generate"))

        # 🔹 Safety check for text
        if not is_safe_text(prompt_en):
            flash("⚠️ Unsafe content detected in your prompt. Please try again.", "danger")
            return redirect(url_for("generate"))

        style_map = {
            "Natural": "photo-realistic, like a real-life photo, highly detailed",
            "Anime": "anime style illustration, colorful",
            "Realism": "highly detailed realistic photography",
            "Oil Painting": "oil painting style, textured brush strokes"
        }

        styled_prompt = f"{style_map.get(style,'')} {prompt_en}"
        flash(f"Generating image with English meaning prompt: '{prompt_en}'", "info")

        try:
            result = pipe(styled_prompt, num_inference_steps=25)
            image = result.images[0]
            gen_fname = f"gen_{uuid.uuid4().hex[:8]}.png"
            gen_full = UPLOADS_DIR / gen_fname
            image.save(str(gen_full))
        except Exception as e:
            flash(f"Image generation failed: {e}", "danger")
            return redirect(url_for("generate"))

        hist = History(
            user_id=current_user.id,
            kind="generate",
            image_filename=gen_fname,
            caption=styled_prompt,
            translated=prompt_en,
            original_input=text_prompt,
            converted=converted
        )
        db.session.add(hist)
        db.session.commit()

        return render_template("generate_result.html",
                               image_filename=gen_fname,
                               prompt_orig=text_prompt,
                               converted_prompt=converted,
                               english_prompt=prompt_en,
                               style=style)

    return render_template("generate.html")

# ---------------- Caption ----------------
@app.route("/caption", methods=["GET", "POST"])
@login_required
def caption():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file uploaded", "danger")
            return redirect(url_for("caption"))
        file = request.files["image"]
        if file.filename == "":
            flash("Empty filename", "danger")
            return redirect(url_for("caption"))

        fname = unique_name(file.filename)
        path = UPLOADS_DIR / fname
        file.save(str(path))

        # 🔹 Safety check for image
        if not is_safe_image(str(path)):
            flash("⚠️ Unsafe image detected. Upload another file.", "danger")
            return redirect(url_for("caption"))

        if CAPTION_AVAILABLE:
            try:
                raw_image = Image.open(path).convert("RGB")
                inputs = processor(raw_image, return_tensors="pt")
                out = blip_model.generate(**inputs)
                caption_en = processor.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                caption_en = f"Captioning failed: {e}"
        else:
            caption_en = "Caption model not available"

        # Translate caption
        target_lang = request.form.get("caption_language", "en")
        try:
            caption_translated = GoogleTranslator(source="en", target=target_lang).translate(caption_en)
        except Exception:
            caption_translated = caption_en

        hist = History(
            user_id=current_user.id,
            kind="caption",
            image_filename=fname,
            caption=caption_en,
            translated=caption_translated
        )
        db.session.add(hist)
        db.session.commit()

        return render_template("caption_result.html",
                               caption=caption_en,
                               caption_translated=caption_translated,
                               image_filename=fname,
                               lang=target_lang)

    return render_template("caption.html")

# ---------------- Speech-to-Text ----------------
@app.route("/speech", methods=["GET", "POST"])
@login_required
def speech():
    if request.method == "POST":
        target_lang = request.form.get("target_lang", "en")
        original_text, translated_text, filename = "", "", ""

        if "audio" not in request.files:
            return render_template("speech.html", error="No audio uploaded.")
        file = request.files["audio"]
        if file.filename == "":
            return render_template("speech.html", error="Empty filename.")
        filename = unique_name(file.filename)
        filepath = UPLOADS_DIR / filename
        file.save(str(filepath))
        wav_path = convert_to_wav(filepath)

        if not wav_path:
            return render_template("speech.html", error="File conversion failed.")

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
            original_text = recognizer.recognize_google(audio)
        except Exception as e:
            original_text = f"Recognition failed: {e}"

        # 🔹 Safety check for speech text
        if not is_safe_text(original_text):
            flash("⚠️ Unsafe speech content detected. Please try again.", "danger")
            return redirect(url_for("speech"))

        try:
            if target_lang == "en":
                translated_text = original_text
            else:
                translated_text = GoogleTranslator(source="auto", target=target_lang).translate(original_text)
        except Exception as e:
            print("Translation error:", e)
            translated_text = "Translation failed"

        hist = History(
            user_id=current_user.id,
            kind="speech",
            image_filename=filename,
            caption=original_text,
            translated=translated_text
        )
        db.session.add(hist)
        db.session.commit()

        return render_template(
            "speech_result.html",
            original_text=original_text,
            translated_text=translated_text,
            filename=filename,
            target_lang=target_lang
        )

    return render_template("speech.html")

# ---------------- History ----------------
@app.route("/history")
@login_required
def history():
    items = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).all()
    return render_template("history.html", history_items=items)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)