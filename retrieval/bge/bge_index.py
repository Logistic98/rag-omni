# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
import torch
import uuid


class BGEIndexer:
    def __init__(self, file_paths, old_index_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'bge-large-zh-v1.5')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.old_index_path = old_index_path
        self.data_list = self.load_data(file_paths)
        self.embeddings_list = self.generate_embeddings()
        self.cuda_oom_flag = False

    def load_data(self, file_paths):
        data_list = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data_list.extend(data)
        return data_list

    def generate_embeddings(self):
        embeddings_list = []
        batch_size = 4
        has_switched_to_cpu = False
        for i in trange(0, len(self.data_list), batch_size):
            batch_texts = [item['part_content'] for item in self.data_list[i:i + batch_size]]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).to('cpu').numpy()
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    if not has_switched_to_cpu:
                        print("CUDA out of memory. Switching to CPU for this batch.")
                        has_switched_to_cpu = True
                    torch.cuda.empty_cache()
                    inputs = inputs.to('cpu')
                    self.model.to('cpu')
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                    self.model.to(self.device)
                else:
                    raise e
            embeddings_list.extend(embeddings)
            torch.cuda.empty_cache()
        return np.array(embeddings_list)

    def _load_old_index(self):
        if not self.old_index_path or not os.path.exists(self.old_index_path):
            return None, None
        data = np.load(self.old_index_path, allow_pickle=True)
        old_embeddings_list = data['embeddings_list']
        old_data_list_json = data['data_list'].item()
        old_data_list = json.loads(old_data_list_json)
        return old_data_list, old_embeddings_list

    def _merge_indexes(self, old_data_list, old_embeddings_list):
        if old_data_list is None or old_embeddings_list is None:
            return self.data_list, self.embeddings_list
        new_data_list = old_data_list + self.data_list
        new_embeddings_list = np.vstack((old_embeddings_list, self.embeddings_list))
        return new_data_list, new_embeddings_list

    def build_index(self, output_path, index_name=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not index_name:
            index_name = str(uuid.uuid4())
        index_file = os.path.join(output_path, f'{index_name}.npz')

        old_data_list, old_embeddings_list = self._load_old_index()
        merged_data_list, merged_embeddings_list = self._merge_indexes(old_data_list, old_embeddings_list)

        data_list_json = json.dumps(merged_data_list, ensure_ascii=False, indent=4)
        np.savez(index_file, embeddings_list=merged_embeddings_list, data_list=data_list_json)
        print(f"Index saved to {index_file}")


if __name__ == '__main__':
    index_name = "bge_index"  # 定义索引名（如果不指定则自动使用uuid生成）
    output_path = "./index"   # 定义索引的存储路径

    # 用一个文件构建初始索引
    file_paths = [
        "../../data/preprocess_data/国务院关于加强地方政府性债务管理的意见.json"
    ]
    indexer = BGEIndexer(file_paths)
    indexer.build_index(output_path, index_name=index_name)

    # 用另一个文件和旧索引增量构建新索引
    file_paths = [
        "../../data/preprocess_data/中共中央办公厅国务院办公厅印发《关于做好地方政府专项债券发行及项目配套融资工作的通知》.json"
    ]
    old_index_path = os.path.join(output_path, f'{index_name}.npz')
    indexer = BGEIndexer(file_paths, old_index_path)
    indexer.build_index(output_path, index_name=index_name)
