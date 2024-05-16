# -*- coding: utf-8 -*-

import math
import os
import jieba
import logging
import json

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


class BM25Algorithm(object):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    _stop_words_path = os.path.join(current_dir, 'stop_words.txt')
    _stop_words = []

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.param: BM25Param = self._load_param()

    def _load_stop_words(self):
        if not os.path.exists(self._stop_words_path):
            raise Exception(f"system stop words: {self._stop_words_path} not found")
        stop_words = []
        with open(self._stop_words_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

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

        param = _cal_param(docs_data)
        return param

    def _load_param(self):
        self._stop_words = self._load_stop_words()
        param = self._build_param()
        return param

    def _cal_similarity(self, words, index):
        score = 0
        for word in words:
            if word not in self.param.f[index]:
                continue
            molecular = self.param.idf[word] * self.param.f[index][word] * (self.param.k1 + 1)
            denominator = self.param.f[index][word] + self.param.k1 * (1 - self.param.b +
                                                                       self.param.b * self.param.line_length_list[
                                                                           index] /
                                                                       self.param.avg_length)
            score += molecular / denominator
        return score

    def search(self, query: str, top_k: int = -1):
        if top_k != -1 and top_k <= 0:
            raise ValueError("top_k should be -1 or a positive integer")

        words = [word for word in jieba.lcut(query) if word and word not in self._stop_words]
        score_list = []
        for index in range(self.param.length):
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))

        score_list.sort(key=lambda x: -x[1])
        if top_k != -1:
            score_list = score_list[:top_k]

        formatted_result = [
            {
                "file_name": os.path.basename(doc["file_path"]).replace('.json', '.docx'),
                "part_content": doc["part_content"],
                "score": score
            }
            for doc, score in score_list
        ]
        return formatted_result


if __name__ == '__main__':
    file_paths = [
        "../../data/preprocess_data/国务院关于加强地方政府性债务管理的意见.json",
        "../../data/preprocess_data/中共中央办公厅国务院办公厅印发《关于做好地方政府专项债券发行及项目配套融资工作的通知》.json"
    ]
    bm25 = BM25Algorithm(file_paths)
    query_content = "国务院对于地方政府性债务管理的意见"
    top_k = 5  # 可以设置为任意正整数，或者-1表示不限制
    result = bm25.search(query_content, top_k)
    print(json.dumps(result, ensure_ascii=False, indent=4))
