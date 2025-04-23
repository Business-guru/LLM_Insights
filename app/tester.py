import requests
import sys
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: python cli_client.py <query> <domain>")
        return

    query = sys.argv[1]
    domain = sys.argv[2]

    url = "http://127.0.0.1:8000/ask"
    payload = {
        "query": query,
        "domain": domain
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        print("\nResponse:")
        print(data['response'])
        print("\nFurther Questions:")
        print(data['further_questions'])
    else:
        print("Error:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    main()
