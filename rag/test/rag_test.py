# -*- coding: utf-8 -*-

import requests
import json

url = "http://127.0.0.1:5002/api/rag/summary"
headers = {
    "Content-Type": "application/json"
}
data = {
    "content": "简要总结一下国家对于地方政府性债务管理的意见",
    "id": "session_id"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
