import sounddevice as sd
import numpy as np
import queue
import sys
from dotenv import load_dotenv
import os
from faster_whisper import WhisperModel
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime  


load_dotenv()  
API_KEY = os.getenv("GROQ_API_KEY")

client_groq = Groq(api_key=API_KEY)

CREDENTIALS_FILE = 'cred.json'
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
gs_client = gspread.authorize(creds)

SPREADSHEET_ID = '1v0zVvH1nF4O-Kxr-WA91_Xj-vdOHsdQ7AS8iaSCsPvM'
sheet = gs_client.open_by_key(SPREADSHEET_ID).sheet1  

samplerate = 16000
blocksize = 1024
q = queue.Queue()

print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded. Speak or play audio. Press Ctrl+C to stop.")

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def analyze_sentiment(text):
    response = client_groq.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
            {"role": "user", "content": f"Classify the sentiment of this text as Positive, Negative, or Neutral and explain in 1-2 sentences why it is so: \"{text}\""}
        ],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content.strip()


def get_sentiment_word(sentiment_full):
    for word in ["Positive", "Negative", "Neutral"]:
        if word.lower() in sentiment_full.lower():
            return word
    return "Neutral"  


def write_to_sheet(transcription, sentiment_full):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    transcription_clean = transcription.replace("\n", " ").strip()
    if transcription_clean == "":
        transcription_clean = "No transcription"

    sentiment_word = get_sentiment_word(sentiment_full)

    next_row = len(sheet.get_all_values()) + 1

    values = [[now, transcription_clean, sentiment_word]]

    sheet.update(f"A{next_row}:C{next_row}", values)
    print(f"Data written to Google Sheet at row {next_row}")

audio_buffer = []

print("Listening... Speak now or play audio. Press Ctrl+C to stop.")

try:
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=blocksize):
        print("Recording... Press Ctrl+C to stop.")
        while True:
            audio_chunk = q.get()
            audio_buffer.append(audio_chunk)

except KeyboardInterrupt:
    print("\nStopped recording, now transcribing...")

    if len(audio_buffer) == 0:
        print("No audio recorded!")
        sys.exit()

    audio_data = np.concatenate(audio_buffer, axis=0).flatten()

    segments, _ = model.transcribe(audio_data, language="en")
    print("\n===== Transcription =====")
    full_text = ""
    for segment in segments:
        print(segment.text)
        full_text += " " + segment.text
   

    print("\n===== Sentiment Analysis =====")
    sentiment_full = analyze_sentiment(full_text.strip())
    print("Sentiment & Explanation:\n", sentiment_full)
    

    write_to_sheet(full_text.strip(), sentiment_full)
