# -*- coding: utf-8 -*-

import argparse
import json
import time
from flask import Flask, jsonify
from flask_cors import CORS
from pre_request import Rule, pre

from rag_solve import LLMService, History, get_knowledge_based_answer
from response import ResponseCode, ResponseMessage
from log import logger

# 解析启动参数
parser = argparse.ArgumentParser(description="启动参数")
parser.add_argument('--api_url', type=str, default="https://api.openai.com/v1/chat/completions", help="LLM API URL")
parser.add_argument('--api_key', type=str, help="LLM API Key")
parser.add_argument('--model', type=str, help="LLM模型名称")
parser.add_argument('--port', type=int, default=5002, help="启动的端口号，默认5002")
parser.add_argument('--retrieval_url', type=str, default="http://127.0.0.1:5001/api/rag/retrieval", help="检索服务的URL")
args = parser.parse_args()

# 初始化参数
api_url = args.api_url
api_key = args.api_key
model = args.model
port = args.port
retrieval_url = args.retrieval_url

# 初始化LLM服务
llm = LLMService(url=api_url, api_key=api_key, model=model)

# 初始化历史消息
session_histories = {}

# 创建一个服务
app = Flask(__name__)
CORS(app, supports_credentials=True)

"""
# 基于RAG的LLM对话服务
"""
@app.route("/api/rag/summary", methods=["POST"])
def get_bot_response():
    global session_histories, llm

    # 获取请求数据
    rule = {
        "content": Rule(type=str, required=True),
        "id": Rule(type=str, required=True)
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

    userText = params["content"]
    session_id = params["id"]

    # 获取对话历史，如果有的话
    if session_id in session_histories:
        history_obj = session_histories[session_id]["history"]
        session_histories[session_id]["last_access_time"] = time.time()
    else:
        history_obj = History()
        session_histories[session_id] = {
            "history": history_obj,
            "last_access_time": time.time(),
        }

    # 如果用户超过一个小时没有交互，则删除该用户的对话历史
    max_idle_time = 60 * 60
    for sid, session_data in session_histories.copy().items():
        idle_time = time.time() - session_data["last_access_time"]
        if idle_time > max_idle_time:
            del session_histories[sid]

    # 清空对话历史
    if userText == "$清空对话历史":
        history_obj.history = []
        success_response = dict(code=ResponseCode.SUCCESS, msg=ResponseMessage.SUCCESS, data="已清空对话历史")
        logger.info(success_response)
        response = jsonify(success_response)
        response.data = json.dumps(success_response, ensure_ascii=False, indent=4)
        return response

    # 获取知识库回答
    try:
        answer = get_knowledge_based_answer(
            query=userText, history_obj=history_obj, url_retrieval=retrieval_url, llm=llm
        )
        success_response = dict(code=ResponseCode.SUCCESS, msg=ResponseMessage.SUCCESS, data=answer)
        logger.info(success_response)
        response = jsonify(success_response)
        response.data = json.dumps(success_response, ensure_ascii=False, indent=4)
        return response
    except Exception as e:
        logger.error(e)
        fail_response = dict(code=ResponseCode.BUSINESS_FAIL, msg=ResponseMessage.BUSINESS_FAIL, data=None)
        logger.error(fail_response)
        response = jsonify(fail_response)
        response.data = json.dumps(fail_response, ensure_ascii=False, indent=4)
        return response


if __name__ == '__main__':
    # 解决中文乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 启动服务，指定主机和端口
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
