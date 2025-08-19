import os
import json
from flask import Flask, render_template_string, request, jsonify
import openai
import pandas as pd

app = Flask(__name__)
app.secret_key = os.getenv("MEDIAID_SECRET", "mediaid-hackathon")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Example static fallback rules
CONDITIONS = [
    {
        "keywords": ["chest pain", "shortness of breath", "severe bleeding", "unconscious", "stroke", "heart attack"],
        "severity": "Severe",
        "advice": "Go to the ER immediately. Call emergency services (911 or your local number)."
    },
    {
        "keywords": ["high fever", "vomiting", "dehydration", "severe headache", "confusion", "stiff neck", "difficulty breathing"],
        "severity": "Moderate",
        "advice": "Consult a doctor within 24 hours. If symptoms worsen, seek emergency care."
    },
    {
        "keywords": ["cough", "runny nose", "mild headache", "mild sore throat", "back pain", "fatigue"],
        "severity": "Mild",
        "advice": "Home care recommended. Rest, stay hydrated, and monitor symptoms."
    }
]

SEVERITY_DISPLAY = {
    "Mild": {"color": "green", "icon": "🏠", "desc": "Home care"},
    "Moderate": {"color": "yellow", "icon": "🩺", "desc": "See a doctor soon"},
    "Severe": {"color": "red", "icon": "🚑", "desc": "Emergency (ER)"}
}

HISTORY_FILE = "mediaid_history.json"
DATASET_FILE = "symptoms_diseases.csv"  # You need to provide this file (see below)

# Load dataset if available
# Example: Dataset should have columns: "symptom", "disease", "risk_level"
def load_dataset():
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        # Clean columns for matching
        df["symptom"] = df["symptom"].str.lower().str.strip()
        df["disease"] = df["disease"].str.strip()
        df["risk_level"] = df["risk_level"].str.strip().str.title()
        return df
    return None

symptoms_df = load_dataset()

def preprocess(text):
    return text.lower().strip()

def dataset_match(symptoms_text):
    if symptoms_df is None:
        return None
    # Try to match any symptom phrase from dataset
    matches = []
    for _, row in symptoms_df.iterrows():
        if row["symptom"] in symptoms_text:
            matches.append(row)
    if not matches:
        return None
    # Take highest risk level if multiple match
    risk_priority = {"Severe": 3, "Moderate": 2, "Mild": 1}
    best = max(matches, key=lambda row: risk_priority.get(row["risk_level"], 0))
    return {
        "disease": best["disease"],
        "severity": best["risk_level"],
        "advice": f"Possible: {best['disease']}. Please consult a doctor for proper diagnosis."
    }

def rule_based_triage(symptoms_text):
    for cond in CONDITIONS:
        for kw in cond["keywords"]:
            if kw in symptoms_text:
                return cond["severity"], cond["advice"]
    return "Mild", "Home care recommended. Rest, stay hydrated, and monitor symptoms."

def llm_triage(symptoms_text):
    if not openai_client:
        return None
    prompt = (
        "You are a medical triage assistant. "
        "A patient describes their symptoms. "
        "Decide the urgency: Mild (home care, self-monitor), Moderate (see a doctor within 24h), Severe (go to ER now). "
        "Give a short summary of severity and advice, and if possible, suggest a likely disease (optional).\n\n"
        f"Symptoms: {symptoms_text}\n"
        "Respond in this exact format:\n"
        "Severity: [Mild/Moderate/Severe]\n"
        "Advice: [brief advice]\n"
        "Disease: [optional, if highly likely]\n"
    )
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        text = completion.choices[0].message.content.strip()
        # Parse result
        severity = "Mild"
        advice = ""
        disease = ""
        for line in text.splitlines():
            if line.lower().startswith("severity:"):
                val = line.split(":", 1)[-1].strip().title()
                if val in SEVERITY_DISPLAY:
                    severity = val
            elif line.lower().startswith("advice:"):
                advice = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("disease:"):
                disease = line.split(":", 1)[-1].strip()
        return severity, advice, disease
    except Exception as e:
        print("LLM error:", e)
        return None

def triage_symptoms(symptoms_text):
    symptoms_text = preprocess(symptoms_text)
    # 1. Try dataset
    dataset_result = dataset_match(symptoms_text)
    if dataset_result:
        return dataset_result["severity"], dataset_result["advice"], dataset_result["disease"]
    # 2. Try LLM
    llm_result = llm_triage(symptoms_text)
    if llm_result:
        severity, advice, disease = llm_result
        return severity, advice, disease
    # 3. Fallback rule-based
    severity, advice = rule_based_triage(symptoms_text)
    return severity, advice, ""

def save_history(entry):
    if not os.path.exists(HISTORY_FILE):
        history = []
    else:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    triage_result = None
    history = load_history()
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "")
        severity, advice, disease = triage_symptoms(symptoms)
        entry = {
            "symptoms": symptoms,
            "severity": severity,
            "advice": advice,
            "disease": disease
        }
        save_history(entry)
        triage_result = entry
        history = load_history()
    return render_template_string(TEMPLATE, triage_result=triage_result, history=history, severity_display=SEVERITY_DISPLAY)

@app.route("/triage", methods=["POST"])
def triage_api():
    data = request.json
    symptoms = data.get("symptoms", "")
    severity, advice, disease = triage_symptoms(symptoms)
    entry = {
        "symptoms": symptoms,
        "severity": severity,
        "advice": advice,
        "disease": disease
    }
    save_history(entry)
    return jsonify(entry)

@app.route("/history")
def history():
    history = load_history()
    return render_template_string(HISTORY_TEMPLATE, history=history, severity_display=SEVERITY_DISPLAY)

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MediAid - AI Triage</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-green-100 to-blue-100 min-h-screen">
  <div class="max-w-xl mx-auto py-10">
    <div class="flex items-center gap-3 mb-8">
      <div class="w-12 h-12 bg-white rounded-xl flex items-center justify-center text-3xl shadow">🩺</div>
      <h1 class="text-4xl font-bold text-blue-700">MediAid</h1>
    </div>
    <div class="bg-white rounded-xl shadow-xl p-8 mb-6">
      <h2 class="text-2xl font-bold mb-3">AI Symptom Triage</h2>
      <form method="POST" class="flex flex-col gap-4">
        <textarea name="symptoms" placeholder="Describe your symptoms, e.g. 'fever and cough for 3 days'" required class="w-full h-24 p-3 rounded-lg border focus:ring-2 focus:ring-blue-500 outline-none"></textarea>
        <button class="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-lg font-bold hover:from-green-600 hover:to-blue-600">Check Severity</button>
      </form>
      {% if triage_result %}
      <div class="mt-8 p-6 rounded-xl shadow-lg border flex flex-col gap-2" style="background: linear-gradient(90deg, {% if severity_display[triage_result.severity].color == 'green' %}#bbf7d0{% elif severity_display[triage_result.severity].color == 'yellow' %}#fef08a{% else %}#fecaca{% endif %}, #fff);">
        <div class="flex items-center gap-3 text-xl font-bold">
          <span class="text-2xl">{{ severity_display[triage_result.severity].icon }}</span>
          <span class="text-gray-700">Severity: </span>
          <span class="text-{{ severity_display[triage_result.severity].color }}-700">{{ triage_result.severity }}</span>
        </div>
        {% if triage_result.disease %}
        <div class="text-lg text-gray-800"><b>Possible Disease:</b> {{ triage_result.disease }}</div>
        {% endif %}
        <div class="text-lg text-gray-800"><b>Advice:</b> {{ triage_result.advice }}</div>
      </div>
      {% endif %}
      <div class="mt-6 text-sm text-gray-500">
        ⚠️ Not medical advice. Always consult a doctor.
      </div>
    </div>
    <div class="bg-white rounded-xl shadow p-6 mt-2">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-lg font-bold">Past Triage History</h3>
        <a href="/history" class="text-blue-500 hover:underline text-sm">Full History</a>
      </div>
      <div class="max-h-48 overflow-y-auto">
      {% for h in history[-5:] %}
        <div class="flex gap-3 mb-2 items-start">
          <span class="mt-1 text-xl">{{ severity_display[h.severity].icon }}</span>
          <div>
            <div class="text-gray-700"><b>Symptoms:</b> {{ h.symptoms }}</div>
            <div class="text-xs text-gray-400"><b>Severity:</b> {{ h.severity }} | <b>Advice:</b> {{ h.advice }}{% if h.disease %} | <b>Disease:</b> {{ h.disease }}{% endif %}</div>
          </div>
        </div>
      {% else %}
        <div class="text-gray-400">No triage history yet.</div>
      {% endfor %}
      </div>
    </div>
    <div class="mt-10 text-xs text-gray-400 text-center">
      Built for the hackathon 🚀 | <a class="underline" href="/history">History</a>
    </div>
  </div>
</body>
</html>
"""

HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MediAid - History</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-green-100 to-blue-100 min-h-screen">
  <div class="max-w-2xl mx-auto py-10">
    <div class="flex items-center gap-3 mb-8">
      <div class="w-12 h-12 bg-white rounded-xl flex items-center justify-center text-3xl shadow">🩺</div>
      <h1 class="text-3xl font-bold text-blue-700">Triage History</h1>
      <a href="/" class="ml-auto text-blue-500 hover:underline text-sm">← Home</a>
    </div>
    <div class="bg-white rounded-xl shadow-xl p-8">
      {% for h in history[::-1] %}
        <div class="mb-5 p-4 rounded-xl shadow border flex gap-4" style="background: linear-gradient(90deg, {% if severity_display[h.severity].color == 'green' %}#bbf7d0{% elif severity_display[h.severity].color == 'yellow' %}#fef08a{% else %}#fecaca{% endif %}, #fff);">
          <span class="text-2xl">{{ severity_display[h.severity].icon }}</span>
          <div>
            <div><b>Symptoms:</b> {{ h.symptoms }}</div>
            <div class="text-xs text-gray-500"><b>Severity:</b> {{ h.severity }} | <b>Advice:</b> {{ h.advice }}{% if h.disease %} | <b>Disease:</b> {{ h.disease }}{% endif %}</div>
          </div>
        </div>
      {% else %}
        <div class="text-gray-400">No triage history available.</div>
      {% endfor %}
    </div>
    <div class="mt-8 text-xs text-gray-400 text-center">
      ⚡ MediAid for hackathon demo
    </div>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)