import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "mistral:7b",
    "prompt": "Translate to French: Nurse",
    "stream": False,
    "options": {
    "temperature": 0.8,
    "num_ctx": 8192
    }
    
}

for i in range(100):
    response = requests.post(url, json=payload)
    print(f"prompt #{i+1}")
    print(response.json()["response"])