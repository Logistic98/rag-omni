# -*- coding: utf-8 -*-

import json
import requests
import random
from tqdm import trange


if __name__ == '__main__':
    url = "http://127.0.0.1:1710/get_bot_response"
    question = ["什么是政府专项债务？", "专项债收入可以用于经常性支出吗？", "政府专项债务应当通过什么偿还？"]
    for i in trange(len(question)):
        data = {"id": random.randint(0, 9999999), "content": question[i]}
        res = requests.post(url, json=data)
        res = json.loads(res.content)
        print("\nQuestion: " + question[i])
        print("\nAnswer: " + res["response"])
        print("\n-------------------------------------------------")
