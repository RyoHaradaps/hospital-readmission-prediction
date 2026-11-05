# src/llm/summarizer.py
"""
LLM Integration for clinical summaries using Local Ollama.
Falls back to a deterministic mock template if Ollama is not running.
"""
import requests
from typing import Dict

SYSTEM_PROMPT_PHYSICIAN = """
You are an expert clinical decision support AI. 
Given a patient's readmission risk score and their top SHAP risk factors, 
write a concise, professional 2-sentence clinical summary for the attending physician. 
Highlight the primary drivers of risk and suggest one actionable intervention.
"""

SYSTEM_PROMPT_PATIENT = """
You are a compassionate, clear healthcare communicator.
Given a patient's readmission risk, write a short, easy-to-understand transition plan (3 bullet points).
Do not use complex medical jargon. Focus on medication adherence, warning signs, and follow-up.
"""

def _generate_mock_summary(risk_score: float, risk_tier: str, factors: Dict[str, float], audience: str) -> str:
    """Fallback template-based summary if Ollama is not available."""
    increasing = [k.replace("_", " ") for k, v in factors.items() if v > 0]
    
    if audience == "physician":
        driver_text = ", ".join(increasing[:2]) if increasing else "no significant risk drivers"
        return (
            f"Patient presents a {risk_score:.1%} ({risk_tier}) risk of 30-day readmission. "
            f"Primary risk drivers include {driver_text}. "
            f"Recommend scheduling a follow-up call within 48 hours and reviewing discharge medications."
        )
    else:
        return (
            f"Your health care team has reviewed your stay. Your risk of returning to the hospital is {risk_score:.1%}.\n"
            f"- Please take all medications exactly as prescribed.\n"
            f"- Watch for symptoms like shortness of breath or chest pain, and call your doctor if they worsen.\n"
            f"- Make sure to attend your follow-up appointment."
        )

def _call_ollama(sys_prompt: str, user_prompt: str) -> str | None:
    """Attempt to call a local Ollama instance."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"{sys_prompt}\n\n{user_prompt}",
                "stream": False
            },
            timeout=5 # Don't hang forever if Ollama isn't running
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass # Ollama not running, silently fail to mock fallback
    return None

def generate_clinical_summary(risk_score: float, risk_tier: str, top_factors: Dict[str, float], audience: str = "physician") -> str:
    """Generate a natural language summary."""
    sys_prompt = SYSTEM_PROMPT_PHYSICIAN if audience == "physician" else SYSTEM_PROMPT_PATIENT
    factors_text = "\n".join([f"- {k}: {'Increases' if v > 0 else 'Decreases'} risk (SHAP: {v:.3f})" for k, v in top_factors.items()])
    user_prompt = f"Patient Risk Score: {risk_score:.1%}\nRisk Tier: {risk_tier}\nTop Factors:\n{factors_text}"

    # 1. Try Ollama (Local)
    ollama_response = _call_ollama(sys_prompt, user_prompt)
    if ollama_response:
        return ollama_response

    # 2. Fallback to Mock Template
    return _generate_mock_summary(risk_score, risk_tier, top_factors, audience)

# --- Quick local test ---
if __name__ == "__main__":
    mock_factors = {"num_medications": -0.78, "charlson_index": 0.45}
    print("--- Physician Summary ---")
    print(generate_clinical_summary(0.45, "Medium", mock_factors, "physician"))
    print("\n--- Patient Summary ---")
    print(generate_clinical_summary(0.45, "Medium", mock_factors, "patient"))