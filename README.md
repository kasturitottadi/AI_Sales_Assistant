# AI Sales Call Assistant  

This project is an **AI-powered sales call assistant**.  
It can analyze customer messages or transcripts (text input only) for **sentiment** and **tone**, and automatically log the results into a **Google Sheet** for easy tracking.  

---

## 🚀 Features
- Real-time **sentiment analysis** (Positive / Negative / Neutral).  
- **Tone detection** (Upset, Friendly, Polite, Neutral, Angry).  
- Works with **text input**.  
- Automatic logging into **Google Sheets** with 4 columns:  
  - Transcript  
  - Sentiment  
  - Tone  
  - Date & Time  

---

## 🛠️ Tech Stack
- **Frontend**: React (with CSS styling)  
- **Backend**: FastAPI (Python)  
- **AI Model**: Groq LLaMA  
- **Database**: Google Sheets (via Service Account API)  

---

## 📥 How to Clone the Project
1. Open your terminal and run:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
⚙️ Backend Setup
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add your environment variables in a .env file:

ini
Copy code
GROQ_API_KEY=your_groq_api_key_here
Add your Google service account file (cred.json) into the backend folder.

Share your Google Sheet with the service account email.

Update SHEET_ID in the backend code with your sheet ID.

Run the backend:

bash
Copy code
uvicorn main:app --reload
🎨 Frontend Setup
Move into the frontend folder:

bash
Copy code
cd frontend
Install dependencies:

bash
Copy code
npm install
Run the React app:

bash
Copy code
npm start
📊 How It Works
User submits text.

The backend sends it to Groq (LLaMA).

AI responds with Sentiment + Tone + Explanation.

The result is stored in memory and appended to Google Sheets.

You can view your analytics/history anytime.

✅ Example Google Sheet Row
Transcript	Sentiment	Tone	Date
I want to cancel my product	Negative	Upset	11/09/2025, 05:50 PM

👨‍💻 Author
Made with ❤️ for learning AI + Web Development.

yaml
Copy code

---

👉 Do you also want me to add the **exact Google Sheets setup guide** (service account, sharing, finding Sheet ID) directly into this README, so someone else could set it up from scratch easily?






You said:
give all in one file to copy 


ChatGPT said:
Perfect 👍 I’ll give you the complete README.md in one file so you can just copy–paste it into your project root.
I’ll also include the Google Sheets setup guide so anyone can set it up from scratch.

✅ Full README.md
markdown
Copy code
# AI Sales Call Assistant  

This project is an **AI-powered sales call assistant**.  
It analyzes customer messages or transcripts (text input only) for **sentiment** and **tone**, and automatically logs the results into a **Google Sheet** for easy tracking.  

---

## 🚀 Features
- Real-time **sentiment analysis** (Positive / Negative / Neutral).  
- **Tone detection** (Upset, Friendly, Polite, Neutral, Angry).  
- Works with **text input**.  
- Automatic logging into **Google Sheets** with 4 columns:  
  - Transcript  
  - Sentiment  
  - Tone  
  - Date & Time  

---

## 🛠️ Tech Stack
- **Frontend**: React (with CSS styling)  
- **Backend**: FastAPI (Python)  
- **AI Model**: Groq LLaMA  
- **Database**: Google Sheets (via Service Account API)  

---

## 📥 How to Clone the Project
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
⚙️ Backend Setup
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Create a .env file in the backend folder and add:

ini
Copy code
GROQ_API_KEY=your_groq_api_key_here
Add your Google service account file cred.json into the backend folder.

Replace the Google Sheet ID in the backend code:

python
Copy code
SHEET_ID = "your_google_sheet_id_here"
Run the backend:

bash
Copy code
uvicorn main:app --reload
🎨 Frontend Setup
Move into the frontend folder:

bash
Copy code
cd frontend
Install dependencies:

bash
Copy code
npm install
Run the React app:

bash
Copy code
npm start
📊 How It Works
User submits text.

The backend sends it to Groq (LLaMA).

AI responds with Sentiment + Tone + Explanation.

The result is stored in memory and appended to Google Sheets.

You can view your analytics/history anytime.

✅ Example Google Sheet Row
Transcript	Sentiment	Tone	Date
I want to cancel my product	Negative	Upset	11/09/2025, 05:50 PM

📗 Google Sheets Setup
Go to Google Cloud Console.

Create a Service Account and download the JSON credentials → save it as cred.json in your backend folder.

Open your Google Sheet → share it with your service account email (something like xxxx@project.iam.gserviceaccount.com) and give Editor access.

Find your Google Sheet ID from the URL:

bash
Copy code
https://docs.google.com/spreadsheets/d/<YOUR_SHEET_ID>/edit#gid=0
Copy the part between /d/ and /edit → that is your Sheet ID.

Put that ID into your backend code:

python
Copy code
SHEET_ID = "YOUR_SHEET_ID"
👨‍💻 Author
Made with ❤️ for learning AI + Web Development.
