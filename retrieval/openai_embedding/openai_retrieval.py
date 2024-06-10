# -*- coding: utf-8 -*-

import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json


class OpenAIRetrieval:
    def __init__(self, index_folder):
        self.embedding_model_params = self.load_embedding_model_params(index_folder)
        self.embedding_model = OpenAIEmbeddings(**self.embedding_model_params)
        self.faiss_vectorstore, self.content, self.metadata = self.load_data(index_folder)

    def load_embedding_model_params(self, index_folder):
        embedding_model_path = os.path.join(index_folder, 'embedding_model_params.pkl')
        if not os.path.exists(embedding_model_path):
            raise FileNotFoundError(f"Embedding model params file not found: {embedding_model_path}")
        with open(embedding_model_path, 'rb') as file:
            embedding_model_params = pickle.load(file)
        return embedding_model_params

    def load_data(self, index_folder):
        embedding_path = os.path.join(index_folder, 'embeddings')
        index_file = os.path.join(embedding_path, 'index.faiss')
        docstore_file = os.path.join(embedding_path, 'docstore.pkl')
        index_to_docstore_id_file = os.path.join(embedding_path, 'index_to_docstore_id.pkl')

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not os.path.exists(docstore_file):
            raise FileNotFoundError(f"Docstore file not found: {docstore_file}")
        if not os.path.exists(index_to_docstore_id_file):
            raise FileNotFoundError(f"Index to docstore ID file not found: {index_to_docstore_id_file}")

        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        with open(docstore_file, 'rb') as f:
            docstore = pickle.load(f)
        with open(index_to_docstore_id_file, 'rb') as f:
            index_to_docstore_id = pickle.load(f)

        embedding_model_instance = OpenAIEmbeddings(**self.embedding_model_params)
        faiss_vectorstore = FAISS(
            index=index,
            embedding_function=embedding_model_instance,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        content_file = os.path.join(index_folder, 'content.pkl')
        metadata_file = os.path.join(index_folder, 'metadata.pkl')

        if not os.path.exists(content_file):
            raise FileNotFoundError(f"Content file not found: {content_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(content_file, 'rb') as file:
            content = pickle.load(file)
        with open(metadata_file, 'rb') as file:
            metadata = pickle.load(file)
        return faiss_vectorstore, content, metadata

    def search(self, query, top_k=5):
        results = self.faiss_vectorstore.similarity_search(query, k=len(self.content))
        # 如果 top_k 为 -1，则返回所有结果
        if top_k == -1:
            top_k = len(results)
        results = results[:top_k]
        search_results = []
        for item in results:
            result = {
                "file_name": item.metadata['file_name'],
                "part_content": item.page_content
            }
            search_results.append(result)
        return search_results


if __name__ == '__main__':
    index_folder = "./index/openai_index"
    query_text = "国务院对于地方政府性债务管理的意见"
    top_k = 5  # 可以设置为任意正整数，或者-1表示不限制
    openai_retriever = OpenAIRetrieval(index_folder)
    results = openai_retriever.search(query_text, top_k=top_k)
    print(json.dumps(results, ensure_ascii=False, indent=4))
