# -*- coding: utf-8 -*-

import requests
import json


class Baichuan:
    def __init__(self, url):
        self.url = url

    def stream_request(self, messages: list):
        data = json.dumps({"messages": messages})
        try:
            with requests.post(self.url, data=data, headers={'Content-Type': 'application/json'}, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_chunk = line.decode('utf-8')
                        yield decoded_chunk
        except requests.RequestException as e:
            print(f"请求错误: {e}")


if __name__ == '__main__':
    llm = Baichuan("http://127.0.0.1:1708")
    messages = [{
        "role": "user",
        "content": "解释一下量子计算"
    }]
    for response in llm.stream_request(messages):
        print(response)
