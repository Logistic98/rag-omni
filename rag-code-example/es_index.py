# -*- coding: utf-8 -*-

import os
from elasticsearch import Elasticsearch
from elasticsearch import helpers

index_name = "policy_qa"
es = Elasticsearch(
    hosts=["http://127.0.0.1:9200"],
    basic_auth=("elastic", "your_es_password"),
    request_timeout=60
)
CREATE_BODY = {
    "settings": {
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "ik_max_word"
            }
        }
    }
}

es.indices.create(index=index_name, body=CREATE_BODY)
directory_path = "./preprocess_data"
contents = []

# 遍历目录下的文件
for filename in os.listdir(directory_path):
    # 确保文件是以txt为扩展名的文本文件
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        # 读取文件内容并添加到列表中
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            contents.append(file_content)

action = (
    {
        "_index": index_name,
        "_type": "_doc",
        "_id": i,
        "_source": {
            "content": contents[i]
        }
    } for i in range(0, len(contents))
)
helpers.bulk(es, action)

print("export es finish")
