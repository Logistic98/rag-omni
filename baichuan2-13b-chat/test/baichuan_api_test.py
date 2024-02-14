# -*- coding: utf-8 -*-


import requests
import json


class Baichuan:
    def __init__(self, url):
        self.url = url

    def __call__(self, messages: list) -> str:
        data = {"messages": messages}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        return response["response"]


if __name__ == '__main__':
    llm = Baichuan("http://127.0.0.1:1707/")
    messages = [{
        "role": "user",
        "content": "解释一下量子计算"
    }]
    response = llm(messages)
    print(response)
