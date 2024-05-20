# -*- coding: utf-8 -*-

import json
import requests


def send_post_request(url, payload):
    """
    向指定的URL发送POST请求。
    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    updated_payload = {
        "model": "qwen-1.5-0.5b",
        "messages": [
            {
                "role": "user",
                "content": payload["prompt"]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "string",
                    "description": "string",
                    "parameters": {}
                }
            }
        ],
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(updated_payload))
    try:
        response_json = response.json()
        print(response_json)
    except ValueError:
        print("Response could not be decoded as JSON:", response.text)


if __name__ == '__main__':
    api_url = 'http://127.0.0.1:5000/v1/chat/completions'
    payload = {
        "prompt": "解释一下量子计算"
    }
    send_post_request(api_url, payload)

