# -*- coding: utf-8 -*-

import argparse
import json
from flask import Flask, jsonify
from flask_cors import CORS
from pre_request import pre, Rule

from log import logger
from response import ResponseCode, ResponseMessage
from bm25_retrieval.bm25 import BM25Algorithm
from bge_retrieval.bge import BGEAlgorithm

# 解析启动参数
parser = argparse.ArgumentParser(description="启动参数")
parser.add_argument('--json_files', type=str, required=True, help="JSON文件路径，多个用逗号分隔")
parser.add_argument('--algorithm', type=str, choices=['BM25', 'BGE'], required=True, help="检索算法：目前仅支持BM25或BGE")
parser.add_argument('--port', type=int, default=5001, help="启动的端口号，默认5001")
args = parser.parse_args()

json_file_paths = args.json_files.split(',')
retrieval_algorithm = args.algorithm
port = args.port

# 创建一个服务
app = Flask(__name__)
CORS(app, supports_credentials=True)

# 加载JSON文件内容到内存中
documents = []
for path in json_file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        documents.extend(json.load(file))

# 初始化检索算法
if retrieval_algorithm == 'BM25':
    search_engine = BM25Algorithm(json_file_paths)
elif retrieval_algorithm == 'BGE':
    search_engine = BGEAlgorithm(json_file_paths)
else:
    raise ValueError("Unsupported retrieval algorithm")

"""
# 检索算法服务
"""
@app.route(rule='/api/rag/retrieval', methods=['GET'])
def retrieval():

    # 参数校验
    rule = {
        "question": Rule(type=str, required=True),
        "top_k": Rule(type=int, required=True, gte=-1, custom=lambda x: x == -1 or x > 0)
    }
    try:
        params = pre.parse(rule=rule)
    except Exception as e:
        logger.error(e)
        fail_response = dict(code=ResponseCode.PARAM_FAIL, msg=ResponseMessage.PARAM_FAIL, data=None)
        logger.error(fail_response)
        response = jsonify(fail_response)
        response.data = json.dumps(fail_response, ensure_ascii=False, indent=4)
        return response

    # 获取参数
    question = params.get("question")
    top_k = params.get("top_k")

    # 业务处理模块
    results = search_engine.search(question, top_k)

    # 成功的结果返回，格式化JSON
    success_response = dict(code=ResponseCode.SUCCESS, msg=ResponseMessage.SUCCESS, data=results)
    logger.info(success_response)
    response = jsonify(success_response)
    response.data = json.dumps(success_response, ensure_ascii=False, indent=4)
    return response


if __name__ == '__main__':
    # 解决中文乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 启动服务，指定主机和端口
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
