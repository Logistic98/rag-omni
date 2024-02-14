# -*- coding: utf-8 -*-

import os
import requests
import json

# Global Parameters
RETRIEVAL_TOP_K = 2
LLM_HISTORY_LEN = 30


class Baichuan:
    def __init__(self, url):
        self.url = url

    def __call__(self, messages: list) -> str:
        data = {"messages": messages}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        return response["response"]


def init_cfg(url_llm):
    global llm
    llm = Baichuan(url=url_llm)


def get_docs(question: str, url: str, top_k=RETRIEVAL_TOP_K):
    data = {"question": question, "top_k": top_k}
    docs = requests.post(url, json=data)
    docs = json.loads(docs.content)
    return docs["docs"]


def get_knowledge_based_answer(query, history_obj, url_retrieval):
    global llm, RETRIEVAL_TOP_K

    if len(history_obj.history) > LLM_HISTORY_LEN:
        history_obj.history = history_obj.history[-LLM_HISTORY_LEN:]

    # Rewrite question
    if len(history_obj.history):
        rewrite_question_input = history_obj.history.copy()
        rewrite_question_input.append(
            {
                "role": "user",
                "content": f"""请基于对话历史，对后续问题进行补全重构，如果后续问题与历史相关，你必须结合语境将代词替换为相应的指代内容，让它的提问更加明确；否则直接返回原始的后续问题。
                注意：请不要对后续问题做任何回答和解释。
                
                后续问题：{query}
                
                修改后的后续问题："""
            }
        )
        new_query = llm(rewrite_question_input)
    else:
        new_query = query

    # 获取相关文档
    docs = get_docs(new_query, url_retrieval, RETRIEVAL_TOP_K)
    doc_string = ""
    for i, doc in enumerate(docs):
        doc_string = doc_string + doc + "\n"
    history_obj.history.append(
        {
            "role": "user",
            "content": f"请基于参考，回答问题，并给出参考依据：\n问题：\n{query}\n参考：\n{doc_string}\n答案："
        }
    )

    # 调用大模型获取回复
    response = llm(history_obj.history)

    # 修改history，将之前的参考资料从history删除，避免history太长
    history_obj.history[-1] = {"role": "user", "content": query}
    history_obj.history.append({"role": "assistant", "content": response})

    # 检查history.json是否存在，如果不存在则创建
    if not os.path.exists("./history.json"):
        with open("./history.json", "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=2)

    # 读取现有数据，追加新数据，并写回文件
    with open("./history.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    data.append({"query": query, "new_query": new_query, "docs": docs, "response": response,
                 "retrieval": "ES"})
    with open("./history.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return {"response": response, "docs": docs}