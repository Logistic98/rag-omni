# -*- coding: utf-8 -*-

import requests
import json


def get_summary(url, user_prompt, session_id):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "user_prompt": user_prompt,
        "session_id": session_id
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


if __name__ == "__main__":
    url = "http://127.0.0.1:5002/api/rag/summary"
    session_id = "session_id_001"

    user_prompt_1 = "简要总结一下国家对于地方政府性债务管理的意见"
    response_1 = get_summary(url, user_prompt_1, session_id)
    print("第一个问题的回复:")
    print(response_1)

    user_prompt_2 = "再详细一些"
    response_2 = get_summary(url, user_prompt_2, session_id)
    print("第二个问题的回复:")
    print(response_2)