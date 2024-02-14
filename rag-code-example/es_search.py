# -*- coding: utf-8 -*-

import json
from flask import Flask, request
from flask_cors import cross_origin
from elasticsearch import Elasticsearch

index_name = "policy_qa"
es = Elasticsearch(
    hosts=["http://127.0.0.1:9200"],
    basic_auth=("elastic", "your_es_password"),
    request_timeout=60
)

app = Flask(__name__)


@app.route('/', methods=['POST'])
@cross_origin()
def retrieval():
    data = json.loads(request.get_data())
    question = data.get("question")
    top_k = data.get("top_k")
    query_body = {
        "query": {
            "match": {
                "content": question
            }
        },
        "size": top_k
    }
    res = es.search(index=index_name, body=query_body)
    docs = []
    for hit in res['hits']['hits']:
        docs.append(hit["_source"]["content"])
    return {"docs": docs}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1709)
