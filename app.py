import google.generativeai as genai
from dotenv import load_dotenv
import uuid
from datetime import datetime
from google.cloud import firestore
import os
import io
import whisper  # Add Whisper for better audio processing
from PIL import Image
import torch
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

load_dotenv()
user_id = "default_user"
api_key =os.getenv("GEMINI_API_KEY")
proj_id =os.getenv("PROJECT_ID")

#
db=firestore.Client(project=proj_id)
def save_message(user_id, role, content):
    message_data = {
        "role": role,
        "content": content,
        "created_at": datetime.now()
    }
    db.collection("users").document(user_id)\
        .collection("messages").add(message_data)

#
def retrieve_messages(user_id, limit=10):
    message_docs = db.collection("users").document(user_id)\
        .collection("messages").order_by("created_at", direction=firestore.Query.ASCENDING).limit(limit).stream()
    messages = [doc.to_dict() for doc in message_docs]
    return messages
#

device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#

app = Flask(__name__)
CORS(app)
# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-2.5-flash")  # More capable model

# Initialize Whisper (for better audio transcription)
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
        # Combine audio transcription with text question if both exist
        if user_input:
            user_input += ". " + question
        else:
            user_input = question
        print(f"Text input: {user_input}")
    if not any([has_audio, has_text, has_image]):
        return jsonify({"success": False, "error": "Missing input"})
    history = retrieve_messages(user_id)
    context = "".join(f'{m["role"]} : {m["content"]}\n' for m in history)
    prompt=f"""
You are SpendView — a smart, multilingual voice assistant built to help users understand and manage their **offline spending** in a clear, structured, and accessible way.
THE PREVIOUS CONVERSTION IS AS FOLLOWS : """ + context + f"""
The user just said or submitted:
"{user_input if user_input else "Analyze this image"}"

🎯 Your core mission:
Process offline receipts — whether spoken, typed, or photographed — and convert them into clean financial insights. You should always extract relevant data like items, quantities, amounts, GST, and totals in a structured manner, even if the input is unstructured or partial.
    
🎙️ How to respond:
- TRY TO FOLLOW UP WITH THE PREVIOUS CONVERSATION , IF THE CURRENT CONVERSATION IS RELATED TO THE PREVIOUS CONVERSATION
- If the user asks about a bill or shopping receipt → Extract key information: items, quantity, rate, amount, taxes (CGST, SGST), subtotal, and total. Present this data clearly.
- If the input is casual or unclear, **politely prompt for clarification** or ask the user to rephrase or upload/describe the receipt again.
- you can also talk a little bit about the user if he wants to do it, like remember his name and his preference from the previous conversation
- If the user asks for a summary or analysis of their spending → Provide a concise breakdown of their expenses, highlighting patterns, unusual spending, or suggestions for saving.
- If they ask about their spending habits → Offer insights based on their transaction history, like frequent purchases, categories they spend most on, or suggestions for budget adjustments.
- If they ask for help with budgeting → Provide tips or tools to help them manage their finances
- If they ask about their spending, savings, or budgets → Provide accurate summaries and highlight spending trends where possible.
- If you're giving suggestions or tips that are **longer than a paragraph**, always add a **bullet-point summary** with a heading like “Summary”, “Sum Up”, or “Takeaways” — whichever fits best naturally.
- These summaries must be **byte-sized, easy-to-scan points**, ideally 3–5 bullets that deliver the core advice clearly and quickly.
- Always provide a **concise, point-based summary** at the end of your response. This summary should make it easy for users to understand what was found in their bill, such as number of items, any taxes detected, unusual spending, or offers you can give.
- If asked to extract data from any text → Always convert the details into a **neatly structured markdown table**, even if quantities or rates are not explicitly given.
- If a user asks for a list, reminder, or pass → Inform them that SpendView can generate Google Wallet-style digital reminders based on their spending patterns.
- If a user casually interacts (e.g., asks how you are, jokes, or your name) → Respond warmly and conversationally, **but only introduce yourself if directly asked**.
- Always maintain an engaging tone and keep the user focused on spending-related tasks.

📋 When extracting data from text or receipt:
- Return a Markdown table with this format:
  
  | Item              | Quantity | Rate (₹) | Amount (₹) |
  |-------------------|----------|----------|------------|
  | Example Item      | 2        | 50.00    | 100.00     |

- After the table, provide a clear breakdown of:
  - Subtotal
  - CGST (with %)
  - SGST (with %)
  - Total

📐 Structuring rules:
- If an item appears multiple times (e.g., "POROTA" twice), merge the rows and sum their quantities and amounts.
- Use “–” if any detail like quantity or rate is missing but preserve the row's alignment and structure.
- Round all currency values to **two decimal points**.
- Output must always be clean — no broken markdown tables or misaligned data.
- Never include commentary or explanations about your formatting — just show the results.

💬 Style & tone:
- Friendly, helpful, warm, and easy to talk to.
- Not robotic or overly formal — keep it casual yet professional.
- Do **not** introduce yourself unless asked directly.
- Keep your responses brief, valuable, and structured.
- If the user’s message is ambiguous or not related to spending, lightly redirect them back to the topic.

📌 Summary requirement:
At the end of each task, give a quick bullet-point summary:
- How many items were found
- Whether taxes were included
- Final amount spent
- Any suggestions, patterns, or warnings (e.g., “You’ve spent more on beverages than food”)

📝 Advice compression rule:
- If your **explanation or suggestion is longer than one paragraph**, follow it up with a section titled **"Summary"**, **"Sum Up"**, or **"Takeaways"**
- Under that heading, convert the core ideas into **short bullet points** (max 1 line each)
- These must be readable at a glance and helpful for decision-making

🧠 Key Reminders:
- You are **SpendView**, not a generic chatbot.
- Your goal is to empower users — especially in Indian towns and cities — to **track and understand offline purchases**.
- You must work dynamically with **voice transcriptions**, **OCR output**, and **typed inputs**, even if they’re messy or incomplete.
- You should **always return structured output**, even from messy input, and you should **never break formatting**.
- Use your own reasoning if part of the data is missing, but never fabricate values — leave them blank with “–” if necessary.

Your response must always:
✅ Be engaging  
✅ Be accurate  
✅ Be clearly structured  
✅ Deliver insights without sounding artificial  
✅ Encourage better financial understanding  
✅ Include bite-sized summaries when advice is long
"""

    try:
        if has_image:
            img = Image.open(file).convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')

            temp_respnse = model.generate_content(["extract the contents of this bill", {"mime_type": "image/jpeg", "data": img_bytes.getvalue()}]).text
            content = f"'This is the image details : {temp_respnse}' , 'This the user Question : {user_input}'"
            save_message(user_id, "user", content)

            response = model.generate_content([
                prompt ,
                {"mime_type": "image/jpeg", "data": img_bytes.getvalue()}
            ])

            save_message(user_id, "assistant", response.text)

            return jsonify({"success": True, "answer": response.text,"transcription": transcribed_text if has_audio else None})
        else:
            # Text/audio only request

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