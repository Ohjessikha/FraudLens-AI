
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"


def generate_explanation(prediction_result):
    """
    Explain an existing ML prediction.
    The language model does not calculate or change the fraud score.
    """

    probability = prediction_result["fraud_probability"]
    prediction = prediction_result["prediction"]
    risk = prediction_result["risk_category"]

    prompt = f"""
You are explaining an educational machine-learning fraud detection prototype.

The machine-learning model returned:
- Estimated fraud probability: {probability:.2%}
- Predicted class: {prediction}
- Demonstration risk category: {risk}

Explain these results in simple English in 3 to 5 short sentences.
Explain that the probability is a model estimate, not proof of fraud.
Mention that a human should verify suspicious transactions.
Do not invent transaction details or claim that you know why a
particular transaction was flagged.
Do not change or recalculate the probability.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        return result.get(
            "response",
            "The AI returned an empty explanation."
        ).strip()

    except requests.exceptions.Timeout:
        return "AI explanation timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is unavailable. Start Ollama and try again. "
            "The ML prediction can still be used."
        )

    except requests.exceptions.RequestException as error:
        return f"Could not generate AI explanation: {error}"


if __name__ == "__main__":
    example_result = {
        "fraud_probability": 0.2134,
        "prediction": "Not flagged",
        "risk_category": "Low",
    }

    print("=== GenAI Explanation Test ===")
    print(generate_explanation(example_result))
