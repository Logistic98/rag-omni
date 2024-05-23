# -*- coding: utf-8 -*-

import requests
import json
import os
import logging
from time import sleep

# 全局参数
RETRIEVAL_TOP_K = 5
LLM_HISTORY_LEN = 30
UNRELATED_RESPONSE = "很抱歉，检索库内不存在与问题相关的参考材料，以下是大模型直接生成的结果："

logging.basicConfig(level=logging.INFO)


class LLMService:
    def __init__(self, url, api_key, model):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model

    def __call__(self, messages: list) -> str:
        data = {
            "model": self.model,
            "messages": messages
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        try:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            logging.error(f"Response content: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            raise


class History:
    def __init__(self, session_id):
        self.session_id = session_id
        self.history = []


def get_docs(question: str, url: str, top_k=RETRIEVAL_TOP_K, retries=3):
    params = {"question": question, "top_k": top_k}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            try:
                docs_response = response.json()
                docs = [doc["part_content"] for doc in docs_response["data"]]
                return docs
            except requests.exceptions.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
                logging.error(f"Response content: {response.text}")
                if attempt < retries - 1:
                    sleep(2 ** attempt)
                else:
                    raise
        except Exception as e:
            logging.error(f"Error in get_docs: {e}")
            if attempt < retries - 1:
                sleep(2 ** attempt)
            else:
                raise


def get_knowledge_based_answer(query, history_obj, url_retrieval, llm):
    global RETRIEVAL_TOP_K, UNRELATED_RESPONSE

    if len(history_obj.history) > LLM_HISTORY_LEN:
        history_obj.history = history_obj.history[-LLM_HISTORY_LEN:]

    # 重构问题
    if len(history_obj.history) > 0:
        rewrite_question_input = history_obj.history.copy()
        rewrite_question_input.append(
            {
                "role": "user",
                "content": f"""请基于对话历史，对后续问题进行补全重构。如果后续问题与历史相关，你必须结合语境将代词替换为相应的指代内容，让它的提问更加明确；否则直接返回原始的后续问题。
                注意：请不要对后续问题做任何回答和解释。

                历史对话：{json.dumps(history_obj.history, ensure_ascii=False)}
                后续问题：{query}

                修改后的后续问题："""
            }
        )
        new_query = llm(rewrite_question_input).strip()
        if "请不要对后续问题做任何回答和解释" in new_query:
            new_query = query
    else:
        new_query = query

    # 获取相关文档
    docs = get_docs(new_query, url_retrieval, RETRIEVAL_TOP_K)
    doc_string = "\n".join([json.dumps(doc, ensure_ascii=False) for doc in docs])

    # 判断文档与重构后的问题是否相关
    relevance_check_input = [
        {"role": "system", "content": "你是一个帮助判断内容是否相关的助手。"},
        {"role": "user", "content": f"问题：{new_query}\n文档：{doc_string}\n请判断这些文档是否与问题相关，如果相关，请返回'相关'，否则返回'无关'。"}
    ]
    relevance_response = llm(relevance_check_input).strip()

    if "无关" in relevance_response:
        # 使用重构的问题调用大模型
        direct_response_input = [{"role": "user", "content": new_query}]
        direct_response = llm(direct_response_input)
        response = f"{UNRELATED_RESPONSE}\n\n{direct_response}"
    else:
        history_obj.history.append(
            {
                "role": "user",
                "content": f"请基于参考，回答问题，并给出参考依据：\n问题：\n{query}\n参考：\n{doc_string}\n答案："
            }
        )
        response = llm(history_obj.history)
        history_obj.history[-1] = {"role": "user", "content": query}
        history_obj.history.append({"role": "assistant", "content": response})

    # 保存history
    current_dir = os.path.dirname(os.path.abspath(__file__))
    history_dir = os.path.join(current_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)
    history_file_path = os.path.join(history_dir, f'history_{history_obj.session_id}.json')

    if not os.path.exists(history_file_path):
        with open(history_file_path, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=2)

    with open(history_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data.append({"query": query, "new_query": new_query, "docs": docs, "response": response})
    with open(history_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return {"response": response, "docs": docs}
