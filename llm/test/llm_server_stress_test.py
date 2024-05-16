# -*- coding: utf-8 -*-

import threading
import requests
import json


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


def threaded_requests(url, payload, num_threads, total_requests):
    """
    创建并启动多线程以达到指定的请求总量。
    """
    rounds = (total_requests + num_threads - 1) // num_threads  # 计算需要的轮数
    for _ in range(rounds):
        threads = []
        for _ in range(num_threads):
            if total_requests <= 0:
                break  # 如果已经达到请求总量，停止创建新线程
            thread = threading.Thread(target=send_post_request, args=(url, payload))
            thread.start()
            threads.append(thread)
            total_requests -= 1

        for thread in threads:
            thread.join()


if __name__ == '__main__':
    api_url = 'http://127.0.0.1:5000/v1/chat/completions'
    payload = {
        "prompt": "解释一下量子计算"
    }
    num_threads = 50       # 线程数
    total_requests = 100   # 总请求数

    threaded_requests(api_url, payload, num_threads, total_requests)