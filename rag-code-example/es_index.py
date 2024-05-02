# -*- coding: utf-8 -*-

import json
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
contents = []

with open("./preprocess_data/preprocess_data.json", "r", encoding="utf-8") as file:
    temp = json.load(file)
contents = contents + temp

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