# -*- coding: utf-8 -*-

import os
from huggingface_hub import snapshot_download

# 设置代理
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 模型仓库的标识
repo_id = "Qwen/Qwen1.5-0.5B"

# 下载模型到指定目录
local_dir = "./Qwen1.5-0.5B"

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

snapshot_download(repo_id=repo_id, local_dir=local_dir)