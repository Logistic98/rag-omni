# -*- coding: utf-8 -*-

import argparse
import logging
from bge.bge_index import BGEIndexer
from bm25.bm25_index import BM25Builder

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="构建索引的参数")
    parser.add_argument('--file_paths', type=str, required=True, help="JSON知识文件路径（多个用逗号分隔）")
    parser.add_argument('--algorithm', type=str, choices=['BM25', 'BGE'], required=True, help="索引算法：目前仅支持BM25或BGE")
    parser.add_argument('--output_path', type=str, required=True, help="索引存储路径")
    parser.add_argument('--index_name', type=str, required=False, help="索引名（可选，如果不指定则自动使用UUID生成）")
    parser.add_argument('--old_index_path', type=str, required=False, help="旧索引路径（可选，传递旧索引则增量构建）")
    args = parser.parse_args()

    file_paths = args.file_paths.split(',')
    algorithm = args.algorithm
    output_path = args.output_path
    index_name = args.index_name
    old_index_path = args.old_index_path

    try:
        if algorithm == 'BGE':
            logging.info("开始构建BGE索引...")
            indexer = BGEIndexer(file_paths, old_index_path)
            indexer.save_index(output_path, index_name)
            logging.info("BGE索引构建成功")
        elif algorithm == 'BM25':
            logging.info("开始构建BM25索引...")
            builder = BM25Builder(file_paths, old_index_path)
            builder.build_index(output_path, index_name)
            logging.info("BM25索引构建成功")
        else:
            raise ValueError("Unsupported algorithm. Please choose either 'BM25' or 'BGE'.")
    except Exception as e:
        logging.error(f"索引构建失败: {e}")
        raise
