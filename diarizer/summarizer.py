import os
import json
import requests

def send_to_mistral(transcript_json):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY environment variable")

    prompt = (
        "Ты — умный и опытный помощник.\n"
        "Твоя задача: составить краткий, но содержательный конспект урока.\n"
        "Главный акцент должен быть на то, что сделал(и) ученик(и) и на ученика в целом\n"
        "Действия учителя нужно максимально сократить в конспекте или вообще убрать, если это не важно в контексте фокуса на ученика\n"
        "Формат:\n"
        "Тема 1 — Важный момент 1; Важный момент 2; ...\n"
        "Тема 2 — Важный момент 1; Важный момент 2; ...\n"
        "Требования:\n"
        "- Игнорируй шум, шутки и не относящиеся к теме реплики.\n"
        "- Определи, где говорит учитель, а где ученики, даже если это не всегда явно.\n"
        "- Группируй идеи по темам, а внутри тем — по смыслу.\n"
        "- Не пиши лишнего, только суть.\n\n"
        f"Вот транскрипт в JSON:\n{json.dumps(transcript_json, ensure_ascii=False)}"
    )

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
