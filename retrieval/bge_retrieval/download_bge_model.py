# -*- coding: utf-8 -*-

import os
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging

# 设置代理
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 配置 transformers 日志
logging.set_verbosity_info()


def download_and_save_model(model_name, save_directory):
    # 下载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # 保存模型和分词器
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"模型和分词器已保存到 {save_directory}")


if __name__ == '__main__':
    model_name = 'BAAI/bge-large-zh-v1.5'
    save_directory = './bge-large-zh-v1.5'
    download_and_save_model(model_name, save_directory)
