# -*- coding: utf-8 -*-

import os
import json
import pickle
import uuid

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class OpenAIIndexer:
    def __init__(self, file_paths, openai_api_base, openai_api_key, embedding_model="text-embedding-3-large", old_index_path=None):
        self.embedding_model = self.create_embedding_model(openai_api_base, openai_api_key, embedding_model)
        self.old_index_path = old_index_path
        self.new_data_list, self.new_content, self.new_metadata = self.load_data(file_paths)

        if self.old_index_path and os.path.exists(self.old_index_path):
            self.old_data_list, self.old_content, self.old_metadata = self.load_existing_data()
            self.data_list = self.old_data_list + self.new_data_list
            self.content = self.old_content + self.new_content
            self.metadata = self.old_metadata + self.new_metadata
        else:
            self.data_list, self.content, self.metadata = self.new_data_list, self.new_content, self.new_metadata

        self.faiss_vectorstore = self.generate_embedding()

    def create_embedding_model(self, openai_api_base, openai_api_key, embedding_model):
        return {
            "openai_api_key": openai_api_key,
            "openai_api_base": openai_api_base,
            "model": embedding_model
        }

    def load_data(self, file_paths):
        """读取数据文件"""
        data_list = []
        content_list = []
        metadata_list = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item['file_name'] = os.path.basename(file_path)
                data_list.append(item)
                content_list.append(item['part_content'])
                metadata_list.append({'file_name': os.path.basename(file_path)})
        return data_list, content_list, metadata_list

    def generate_embedding(self):
        embedding_model_instance = OpenAIEmbeddings(**self.embedding_model)
        return FAISS.from_texts(self.content, embedding_model_instance, metadatas=self.metadata)

    def get_index_folder(self, output_path, index_name):
        return os.path.join(output_path, index_name)

    def load_existing_data(self):
        index_folder = self.old_index_path
        embedding_path = os.path.join(index_folder, 'embeddings')

        with open(os.path.join(embedding_path, 'index.faiss'), 'rb') as f:
            index = pickle.load(f)

        with open(os.path.join(embedding_path, 'docstore.pkl'), 'rb') as f:
            docstore = pickle.load(f)

        with open(os.path.join(embedding_path, 'index_to_docstore_id.pkl'), 'rb') as f:
            index_to_docstore_id = pickle.load(f)

        embedding_model_instance = OpenAIEmbeddings(**self.embedding_model)
        faiss_vectorstore = FAISS(
            index=index,
            embedding_function=embedding_model_instance,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        with open(os.path.join(index_folder, 'content.pkl'), 'rb') as file:
            content = pickle.load(file)

        with open(os.path.join(index_folder, 'metadata.pkl'), 'rb') as file:
            metadata = pickle.load(file)

        # 重新构造数据列表
        data_list = [{'part_content': c, 'metadata': m} for c, m in zip(content, metadata)]
        return data_list, content, metadata

    def build_index(self, output_path, index_name=None):
        if not index_name:
            index_name = str(uuid.uuid4())
        index_folder = self.get_index_folder(output_path, index_name)
        os.makedirs(index_folder, exist_ok=True)
        embedding_path = os.path.join(index_folder, 'embeddings')
        os.makedirs(embedding_path, exist_ok=True)

        with open(os.path.join(embedding_path, 'index.faiss'), 'wb') as f:
            pickle.dump(self.faiss_vectorstore.index, f)

        with open(os.path.join(embedding_path, 'docstore.pkl'), 'wb') as f:
            pickle.dump(self.faiss_vectorstore.docstore, f)

        with open(os.path.join(embedding_path, 'index_to_docstore_id.pkl'), 'wb') as f:
            pickle.dump(self.faiss_vectorstore.index_to_docstore_id, f)

        with open(os.path.join(index_folder, 'embedding_model_params.pkl'), 'wb') as file:
            pickle.dump(self.embedding_model, file)

        with open(os.path.join(index_folder, 'content.pkl'), 'wb') as file:
            pickle.dump(self.content, file)

        with open(os.path.join(index_folder, 'metadata.pkl'), 'wb') as file:
            pickle.dump(self.metadata, file)

        print(f"Index saved to {index_folder}")


if __name__ == '__main__':
    index_name = "openai_index"  # 定义索引名（如果不指定则自动使用uuid生成）
    output_path = "./index"  # 定义索引的存储路径
    openai_api_base = "https://api.openai.com/v1"
    openai_api_key = "sk-xxx"

    # 用一个文件构建初始索引
    file_paths = [
        "../../data/preprocess_data/国务院关于加强地方政府性债务管理的意见.json"
    ]
    indexer = OpenAIIndexer(file_paths, openai_api_base, openai_api_key)
    indexer.build_index(output_path, index_name)

    # 用另一个文件和旧索引增量构建新索引
    new_file_paths = [
        "../../data/preprocess_data/中共中央办公厅国务院办公厅印发《关于做好地方政府专项债券发行及项目配套融资工作的通知》.json"
    ]
    old_index_path = os.path.join(output_path, index_name)
    indexer = OpenAIIndexer(new_file_paths, openai_api_base, openai_api_key, old_index_path=old_index_path)
    indexer.build_index(output_path, index_name)
