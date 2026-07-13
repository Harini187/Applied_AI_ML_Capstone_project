# 🛡️Part 4: Secure Production LLM Gateway

## Overview
This repository contains a robust, production-ready LLM Gateway built with Streamlit. It assesses real estate risk based on the Ames Housing dataset features using OpenAI's `gpt-4o-mini` model.

## Core Features Implemented
1. **Prompt Engineering:** Uses a strict System Prompt with a Rubric and Worked Example (Few-shot prompting) to guide the LLM.
2. **Structured Outputs:** Forces the LLM to output pure JSON.
3. **Validation & Fallbacks:** Implements `jsonschema` to strictly validate the LLM's output against a predefined template. If the LLM hallucinates or returns invalid formatting, a safe fallback mechanism is triggered.
4. **PII Guardrails:** Includes regex-based scanning to detect Personally Identifiable Information (like emails) to prevent sensitive data leakage.
5. **Secure Configuration:** Uses `.env` (environment variables) to ensure API keys are never hardcoded or leaked into version control.

## How to Run Locally
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install streamlit requests jsonschema python-dotenv
  streamlit run app.py
  http://localhost:8501
