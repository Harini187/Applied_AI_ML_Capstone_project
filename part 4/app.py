import streamlit as st
import os
import requests
import json
import re
import jsonschema
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.environ.get("LLM_API_KEY")
OPENROUTER_URL = "https://api.openai.com/v1/chat/completions"

# LLM Function
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if not API_KEY: return "Error: API Key missing"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Exception: {e}"

# PII Guardrail
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    return bool(re.search(email_pattern, text))

# Schema & Fallback
housing_assessment_schema = {
    "type": "object",
    "properties": {
        "risk_tier": {"type": "string", "enum": ["low", "medium", "high"]},
        "flag_for_review": {"type": "boolean"},
        "primary_signal": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_action": {"type": "string"}
    },
    "required": ["risk_tier", "flag_for_review", "primary_signal", "confidence", "recommended_action"]
}

fallback_assessment = {
    "risk_tier": "low", "flag_for_review": True, 
    "primary_signal": "N/A", "confidence": "low", "recommended_action": "Manual Review"
}

# System Prompt
system_prompt_housing = """You are a real estate risk engine. Output ONLY raw JSON.
Rubric: High Risk (Year Built < 1950, Qual < 5); Medium Risk (Mismatch); Low Risk (Built > 2000, Qual >= 6).
Worked Example: {"Gr Liv Area": 950, "Year Built": 1940, "Overall Qual": 4} -> 
{"risk_tier": "high", "flag_for_review": true, "primary_signal": "Old construction", "confidence": "high", "recommended_action": "Structural appraisal"}"""

ames_records = [
    {"Gr Liv Area": 1656, "Year Built": 1998, "Overall Qual": 6},
    {"Gr Liv Area": 2800, "Year Built": 2006, "Overall Qual": 9},
    {"Gr Liv Area": 900, "Year Built": 1935, "Overall Qual": 3}
]

# UI
st.title("🛡️ Part 4: LLM Gateway")
if st.button("Run Batch Scoring"):
    for record in ames_records:
        st.json(record)
        raw = call_llm(system_prompt_housing, json.dumps(record), temperature=0.0)
        try:
            parsed = json.loads(raw.strip())
            jsonschema.validate(instance=parsed, schema=housing_assessment_schema)
            st.success("Valid JSON")
            st.json(parsed)
        except:
            st.error("Validation failed")
            st.json(fallback_assessment)