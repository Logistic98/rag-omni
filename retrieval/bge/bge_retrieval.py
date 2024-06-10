# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss


class BGERetrieval:
    def __init__(self, index_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'bge-large-zh-v1.5')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.data_list, self.embeddings_list = self.load_index(index_file)
        self.faiss_index = self.build_faiss_index()

    def load_index(self, index_file):
        data = np.load(index_file, allow_pickle=True)
        embeddings_list = data['embeddings_list']
        data_list_json = data['data_list'].item()
        data_list = json.loads(data_list_json)
        return data_list, embeddings_list

    def build_faiss_index(self):
        faiss_index = faiss.IndexFlatIP(self.embeddings_list.shape[1])
        faiss_index.add(self.embeddings_list)
        return faiss_index

    def search(self, query, top_k=-1):
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1).to('cpu').numpy()
        if top_k == -1:
            top_k = len(self.data_list)
        score, rank = self.faiss_index.search(query_emb, top_k)
        rank = rank[0]
        score = score[0]
        results = [
            {
                "file_name": self.data_list[rank[i]]["file_name"],
                "part_content": self.data_list[rank[i]]["part_content"],
                "score": float(score[i])
            }
            for i in range(top_k)
        ]
        return results


if __name__ == '__main__':
    index_file = "./index/bge_index.npz"
    query_text = "国务院对于地方政府性债务管理的意见"
    top_k = -1  # 可以设置为任意正整数，或者-1表示不限制
    retriever = BGERetrieval(index_file)
    results = retriever.search(query_text, top_k)
    print(json.dumps(results, ensure_ascii=False, indent=4))
