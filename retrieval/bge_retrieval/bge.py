# -*- coding: utf-8 -*-

import os
import faiss
import json
import numpy as np
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
import torch


class BGEAlgorithm:
    def __init__(self, file_paths):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'bge-large-zh-v1.5')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.data_list = self.load_data(file_paths)
        self.embeddings_list = self.generate_embeddings()
        self.faiss_index = self.build_faiss_index()

    def load_data(self, file_paths):
        """读取数据文件并生成嵌入"""
        data_list = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item['file_path'] = file_path
                data_list.append(item)
        return data_list

    def generate_embeddings(self):
        """生成嵌入"""
        embeddings_list = []
        batch_size = 32
        for i in trange(0, len(self.data_list), batch_size):
            batch_texts = [item['part_content'] for item in self.data_list[i:i + batch_size]]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_list.extend(embeddings)
        return embeddings_list

    def build_faiss_index(self):
        """构建Faiss索引"""
        doc_embeddings = np.array(self.embeddings_list)
        faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        return faiss_index

    def search(self, query, top_k=-1):
        """检索函数"""
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        if top_k == -1:
            top_k = len(self.data_list)
        score, rank = self.faiss_index.search(query_emb, top_k)
        rank = rank[0]
        score = score[0]
        results = [
            {
                "file_name": os.path.basename(self.data_list[rank[i]]["file_path"]).replace('.json', '.docx'),
                "part_content": self.data_list[rank[i]]["part_content"],
                "score": float(score[i])
            }
            for i in range(top_k)
        ]
        return results


if __name__ == '__main__':
    file_paths = [
        "../../data/preprocess_data/国务院关于加强地方政府性债务管理的意见.json",
        "../../data/preprocess_data/中共中央办公厅国务院办公厅印发《关于做好地方政府专项债券发行及项目配套融资工作的通知》.json"
    ]
    query_text = "国务院对于地方政府性债务管理的意见"
    top_k = 5  # 可以设置为任意正整数，或者-1表示不限制
    bge = BGEAlgorithm(file_paths)
    results = bge.search(query_text, top_k)
    print(json.dumps(results, ensure_ascii=False, indent=4))
