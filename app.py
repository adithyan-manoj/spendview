import google.generativeai as genai
from dotenv import load_dotenv
import uuid
from datetime import datetime
from google.cloud import firestore
import os
import io
import whisper
from PIL import Image
import torch
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

# Load all secrets from .env
load_dotenv()
user_id = "default_user"

# Read Gemini + Firestore environment variables
api_key = os.getenv("GEMINI_API_KEY")
proj_id = os.getenv("PROJECT_ID")

# Load service account credentials from .env
credentials_dict = {
    "type": os.getenv("TYPE"),
    "project_id": proj_id,
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_CERT_URL"),
    "universe_domain": os.getenv("UNIVERSE_DOMAIN")
}

# Authenticate Firestore
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
db = firestore.Client(project=proj_id, credentials=credentials)

def save_message(user_id, role, content):
    message_data = {
        "role": role,
        "content": content,
        "created_at": datetime.now()
    }
    db.collection("users").document(user_id).collection("messages").add(message_data)

def retrieve_messages(user_id, limit=10):
    message_docs = db.collection("users").document(user_id)\
        .collection("messages").order_by("created_at", direction=firestore.Query.ASCENDING).limit(limit).stream()
    messages = [doc.to_dict() for doc in message_docs]
    return messages

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# Load Whisper
whisper_model = whisper.load_model("base").to(device)

@app.route('/')
def index():
    return render_template('index.html', answer1=None)

@app.route('/que', methods=['POST'])
def ai_response():
    file = request.files.get('bill')
    audio_file = request.files.get('audio')
    question = request.form.get('question')

    transcribed_text = ""
    user_input = ""
    has_audio = audio_file and audio_file.filename != ''
    has_image = file and file.filename != ''
    has_text = question and question.strip() != ''

    if has_audio:
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)
        result = whisper_model.transcribe(audio_path, fp16=False)
        transcribed_text = result['text']
        user_input = transcribed_text
        print(f"Audio input: {user_input}")

    if has_text:
        user_input = f"{user_input}. {question}" if user_input else question
        print(f"Text input: {user_input}")

    if not any([has_audio, has_text, has_image]):
        return jsonify({"success": False, "error": "Missing input"})

    history = retrieve_messages(user_id)
    context = "".join(f'{m["role"]} : {m["content"]}\n' for m in history)

    prompt = f"""
You are SpendView — a smart, multilingual voice assistant built to help users understand and manage their **offline spending** in a clear, structured, and accessible way.
THE PREVIOUS CONVERSTION IS AS FOLLOWS : {context}
The user just said or submitted:
"{user_input if user_input else "Analyze this image"}"
... [the rest of your prompt here, same as provided above]
"""  # TRUNCATED FOR BREVITY

    try:
        if has_image:
            img = Image.open(file).convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')

            temp_response = model.generate_content([
                "extract the contents of this bill",
                {"mime_type": "image/jpeg", "data": img_bytes.getvalue()}
            ]).text

            content = f"This is the image details: {temp_response} | User Question: {user_input}"
            save_message(user_id, "user", content)

            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes.getvalue()}
            ])

            save_message(user_id, "assistant", response.text)

            return jsonify({
                "success": True,
                "answer": response.text,
                "transcription": transcribed_text if has_audio else None
            })
        else:
            save_message(user_id, "user", user_input)
            response = model.generate_content(prompt)
            save_message(user_id, "assistant", response.text)
            return jsonify({
                "success": True,
                "answer": response.text,
                "transcription": transcribed_text if has_audio else None
            })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))
