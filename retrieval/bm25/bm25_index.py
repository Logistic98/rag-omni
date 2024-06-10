# -*- coding: utf-8 -*-

import math
import os
import jieba
import logging
import json
import uuid

jieba.setLogLevel(log_level=logging.INFO)


class BM25Param(object):
    def __init__(self, f, df, idf, length, avg_length, docs_list, line_length_list, k1=1.5, k2=1.0, b=0.75):
        self.f = f
        self.df = df
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

    def __str__(self):
        return f"k1:{self.k1}, k2:{self.k2}, b:{self.b}"


class BM25Indexer(object):
    def __init__(self, file_paths, old_index_path=None):
        self.file_paths = file_paths
        self.old_index_path = old_index_path
        self._stop_words = self._load_stop_words()

    def _load_stop_words(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stop_words_path = os.path.join(current_dir, 'stop_words.txt')
        if not os.path.exists(stop_words_path):
            raise Exception(f"system stop words: {stop_words_path} not found")
        stop_words = []
        with open(stop_words_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _load_old_index(self):
        if not self.old_index_path or not os.path.exists(self.old_index_path):
            return None
        with open(self.old_index_path, 'r', encoding='utf8') as f:
            old_index_data = json.load(f)
        return BM25Param(**old_index_data)

    def _merge_indexes(self, old_param, new_param):
        if not old_param:
            return new_param

        combined_length = old_param.length + new_param.length
        combined_avg_length = (
            (old_param.avg_length * old_param.length) + (new_param.avg_length * new_param.length)
        ) / combined_length

        for word, freq in new_param.df.items():
            if word in old_param.df:
                old_param.df[word] += freq
            else:
                old_param.df[word] = freq

        for word, score in new_param.idf.items():
            if word in old_param.idf:
                old_param.idf[word] = (old_param.idf[word] * old_param.length + score * new_param.length) / combined_length
            else:
                old_param.idf[word] = score

        old_param.f.extend(new_param.f)
        old_param.docs_list.extend(new_param.docs_list)
        old_param.line_length_list.extend(new_param.line_length_list)

        old_param.length = combined_length
        old_param.avg_length = combined_avg_length

        return old_param

    def _build_param(self):
        def _cal_param(docs_data):
            f = []
            df = {}
            idf = {}
            length = len(docs_data)
            words_count = 0
            docs_list = []
            line_length_list = []
            for doc in docs_data:
                content = doc.get("part_content", "").strip()
                if not content:
                    continue
                words = [word for word in jieba.lcut(content) if word and word not in self._stop_words]
                line_length_list.append(len(words))
                docs_list.append(doc)
                words_count += len(words)
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                f.append(tmp_dict)
                for word in tmp_dict.keys():
                    df[word] = df.get(word, 0) + 1
            for word, num in df.items():
                idf[word] = math.log((length - num + 0.5) / (num + 0.5) + 1)
            param = BM25Param(f, df, idf, length, words_count / length, docs_list, line_length_list)
            return param

        docs_data = []
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                raise Exception(f"input docs {file_path} not found")
            with open(file_path, 'r', encoding='utf8') as reader:
                docs = json.load(reader)
                for doc in docs:
                    doc["file_path"] = file_path
                docs_data.extend(docs)

        new_param = _cal_param(docs_data)
        old_param = self._load_old_index()

        return self._merge_indexes(old_param, new_param)

    def build_index(self, output_path, index_name=None):
        param = self._build_param()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not index_name:
            index_name = str(uuid.uuid4())
        index_file = os.path.join(output_path, f'{index_name}.json')
        with open(index_file, 'w', encoding='utf8') as f:
            json.dump(param.__dict__, f, ensure_ascii=False, indent=4)
        print(f"Index saved to {index_file}")


if __name__ == '__main__':

    index_name = "bm25_index"      # 定义索引名（如果不指定则自动使用uuid生成）
    output_path = "./index"        # 定义索引的存储路径

    # 用一个文件构建初始索引
    file_paths = [
        "../../data/preprocess_data/国务院关于加强地方政府性债务管理的意见.json"
    ]
    indexer = BM25Indexer(file_paths)
    indexer.build_index(output_path, index_name=index_name)

    # 用另一个文件和旧索引增量构建新索引
    file_paths = [
        "../../data/preprocess_data/中共中央办公厅国务院办公厅印发《关于做好地方政府专项债券发行及项目配套融资工作的通知》.json"
    ]
    old_index_path = "{}/{}.json".format(output_path, index_name)
    indexer = BM25Indexer(file_paths, old_index_path)
    indexer.build_index(output_path, index_name=index_name)
