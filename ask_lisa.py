import requests

api_token = None
url = "https://chat-1.ki-awz.iisys.de/api/chat/completions"

with open("api_token.txt", encoding="UTF-8") as rf:
    api_token = rf.read().strip()


def ask_lisa(system_prompt: str, question: str) -> str:
    global url
    global api_token

    model: str = "qwen/qwen3-next-80b"
    headers = {'Authorization': f'Bearer {api_token}',
               "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    }
    response = requests.post(url, json=data, headers=headers)

    print("Response status code:", response.status_code)

    response_json = response.json()

    data = response_json["choices"][0]["message"]["content"]

    return str(data)
