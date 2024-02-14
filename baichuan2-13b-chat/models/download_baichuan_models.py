# -*- coding: utf-8 -*-

import os
from huggingface_hub import snapshot_download

# 模型仓库的标识
repo_id = "baichuan-inc/Baichuan2-13B-Chat"

# 下载模型到指定目录
local_dir = "./{}".format(repo_id)

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

snapshot_download(repo_id=repo_id, local_dir=local_dir)