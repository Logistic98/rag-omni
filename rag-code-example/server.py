# -*- coding: utf-8 -*-

from flask import Flask, request
from flask_cors import cross_origin
import time
from solve import *

app = Flask(__name__)


class History:
    def __init__(self):
        self.history = []


session_histories = {}


@app.route("/get_bot_response", methods=["POST"])
@cross_origin()
def get_bot_response():
    global session_histories
    data = json.loads(request.get_data())
    userText = data["content"]  # 用户输入
    session_id = data["id"]  # 用户id，用于保存对话历史

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
    for session_id, session_data in session_histories.copy().items():
        idle_time = time.time() - session_data["last_access_time"]
        if idle_time > max_idle_time:
            del session_histories[session_id]

    if userText == "清空对话历史":
        history_obj.history = []
        return str("已清空")

    response = get_knowledge_based_answer(
        query=userText, history_obj=history_obj, url_retrieval="http://127.0.0.1:1709/"
    )
    return response


if __name__ == "__main__":
    init_cfg("http://127.0.0.1:1707/")
    app.run(host="0.0.0.0", port=1710)
