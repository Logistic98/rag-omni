# 基于python3.9镜像创建新镜像
FROM python:3.9
# 创建容器内部目录
RUN mkdir /code
# 将项目复制到内部目录
ADD . /code/
# 切换到工作目录
WORKDIR /code
# 安装项目依赖
RUN pip install -r requirements.txt
# 安装vim命令
RUN apt-get update && apt-get install vim -y  