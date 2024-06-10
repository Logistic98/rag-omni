# -*- coding: utf-8 -*-

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


class BM25Retrieval(object):
    def __init__(self, index_path):
        self.index_path = index_path
        self.param: BM25Param = self._load_param()
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

    def _load_param(self):
        if not os.path.exists(self.index_path):
            raise Exception(f"Index file {self.index_path} not found")
        with open(self.index_path, 'r', encoding='utf8') as f:
            data = json.load(f)
            param = BM25Param(**data)
        param.length = len(param.f)
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
        for index in range(len(self.param.f)):
            if index >= len(self.param.f):
                raise IndexError(f"Index {index} is out of range for parameter f")
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))

        score_list.sort(key=lambda x: -x[1])
        if top_k != -1:
            score_list = score_list[:top_k]

        result = [
            {
                "file_name": doc["file_name"],
                "part_content": doc["part_content"],
                "score": score
            }
            for doc, score in score_list
        ]
        return result


if __name__ == '__main__':
    index_path = "./index/bm25_index.json"
    bm25 = BM25Retrieval(index_path)
    query_content = "国务院对于地方政府性债务管理的意见"
    top_k = 5  # 可以设置为任意正整数，或者-1表示不限制
    result = bm25.search(query_content, top_k)
    print(json.dumps(result, ensure_ascii=False, indent=4))
