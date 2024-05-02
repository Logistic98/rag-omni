## 1. 检索增强生成

### 1.1 RAG基本介绍

#### 1.1.1 RAG是什么

开源的基座模型参数量不够大，本身拥有的能力有限。要完成复杂的知识密集型的任务，可以基于语言模型构建一个系统，通过访问外部知识源来做到。这样可以使生成的答案更可靠，有助于缓解“幻觉”问题。

RAG 会接受输入并检索出一组相关/支撑的文档，并给出文档的来源。这些文档作为上下文和输入的原始提示词组合，送给文本生成器得到最终的输出。这样 RAG 更加适应事实会随时间变化的情况，这非常有用，因为 LLM 的参数化知识是静态的，RAG 让语言模型不用重新训练就能够获取最新的信息，基于检索生成产生可靠的输出。

![RAG基本介绍](README.assets/RAG基本介绍.png)

#### 1.1.2 RAG发展历程

“RAG”概念由Lewis在2020年引入，其发展迅速，标志着研究旅程中的不同阶段。最初，这项研究旨在通过在预训练阶段为它们注入额外知识来增强语言模型。ChatGPT的推出引发了对利用大型模型进行深度上下文理解的高度兴趣，加速了RAG在推断阶段的发展。随着研究人员更深入地探索大型语言模型（LLMs）的能力，焦点转向提升他们的可控性和推理技巧以跟上日益增长的需求。GPT-4 的出现标志着一个重要里程碑，它革新了 RAG ，采取一种将其与微调技术相结合的新方法，并继续优化预训练策略。

![RAG发展时间轴](README.assets/RAG发展时间轴.png)

#### 1.1.3 RAG生态及挑战

RAG的应用已不再局限于问答系统，其影响力正在扩展到更多领域。现在，诸如推荐系统、信息提取和报告生成等各种任务开始从RAG技术的应用中受益。与此同时，RAG技术栈正在经历一次繁荣。除了众所周知的工具如Langchain和LlamaIndex外，市场上也出现了更多针对性强的RAG工具，例如：为满足更专注场景需求而定制化的；为进一步降低入门门槛而简化使用的；以及功能专业化、逐渐面向生产环境目标发展的。

RAG当前面临的挑战：

- 上下文长度：当检索到的内容过多并超出窗口限制时该怎么办？如果LLMs的上下文窗口不再受限，应如何改进RAG？
- 鲁棒性：如何处理检索到的错误内容？如何筛选和验证检索到的内容？如何增强模型对毒化和噪声的抵抗力？
- 与微调协同工作：如何同时利用RAG和FT的效果，它们应该如何协调、组织，是串行、交替还是端对端？
- 规模定律：RAG模型是否满足规模定律？会有什么情况下可能让RAG经历逆向规模定律现象呢？
- 生产环境应用：如何减少超大规模语料库的检索延迟? 如何确保被 LLMS 检索出来的内容不会泄露?

### 1.2 RAG技术实现

#### 1.2.1 RAG技术范式

在RAG的技术发展中，我们从技术范式的角度总结了其演变过程，主要分为以下几个阶段：

- 初级RAG：初级RAG主要包括三个基本步骤：1）索引——将文档语料库切分成更短的片段，并通过编码器建立向量索引。2）检索——根据问题和片段之间的相似性检索相关文档片段。3）生成——依赖于检索到的上下文来生成对问题的回答。
- 高级RAG：初级RAG在检索、生成和增强方面面临多重挑战。随后提出了高级RAG范式，涉及到预检索和后检索阶段额外处理。在检索之前，可以使用查询重写、路由以及扩展等方法来调整问题与文档片段之间语义差异。在检索之后，重新排列已获取到的文档语料库可以避免"迷失在中间"现象，或者可以过滤并压缩上下文以缩短窗口长度。
- 模块化RAG：随着RAG技术进一步发展和演变，模块化RAG的概念诞生了。结构上，它更自由灵活，引入更具体功能模块如查询搜索引擎以及多答案融合。技术层面上，它将信息查找与微调、强化学习等技术集成起来。在流程方面，RAG模块设计并协同工作形成各种不同类型RAG。

然而，模块化 RAG 并非突然出现，这三种范式存在继承与发展关系。高级RAG是模块化RAG的特殊情况，而初级RAG是高级RAG的特殊情况。

![RAG技术范式](README.assets/RAG技术范式.png)

#### 1.2.2 RAG基本流程

基本流程概述：用户输入问题——>问题重构（补全指代信息，保证多轮对话的能力）——>从检索库检索答案——用LLM总结答案

RAG 由两部分组成：

- 第一部分负责在知识库中，根据 query 检索出匹配的文档。
- 第二部分将 query 和文档拼接起来作为 QA 的 prompt，送入 seq2seq 模型，生成回复。

![RAG原理](README.assets/RAG原理.png)

#### 1.2.3 选择RAG还是微调

除了RAG之外，LLMs的主要优化策略还包括提示工程和微调（FT）。每种都有其独特的特点。根据它们对外部知识的依赖性以及对模型调整的需求，每种都有适合的应用场景。

![RAG与FT的比较](README.assets/RAG与FT的比较.jpg)

RAG就像是给模型提供了一本定制信息检索的教科书，非常适合特定的查询。另一方面，FT就像一个学生随着时间内化知识，更适合模仿特定的结构、风格或格式。通过增强基础模型的知识、调整输出和教授复杂指令，FT可以提高模型的性能和效率。然而，它并不擅长整合新知识或快速迭代新用例。RAG和FT并不互斥，它们相辅相成，并且同时使用可能会产生最好的结果。

![RAG与FT的关系](README.assets/RAG与FT的关系.png)

#### 1.2.4 如何评价RAG的效果

对RAG的评估方法多种多样，主要包括三个质量分数：上下文相关性、答案准确性和答案相关性。此外，评估还涉及四项关键能力：抗噪声能力、拒绝能力、信息整合以及反事实鲁棒性。这些评价维度将传统的定量指标与针对RAG特点的专门评估标准相结合，尽管这些标准尚未得到标准化。

在评价框架方面，有RGB和RECALL等基准测试，以及像RAGAS、ARES和TruLens等自动化评价工具，它们帮助全面衡量RAG模型的表现。

![如何评价RAG的效果](README.assets/如何评价RAG的效果.png)

## 2. 部署大模型服务

实验环境：租用的AutoDL的GPU服务器，NVIDIA RTX 4090 / 24GB，Ubuntu20.04，Python 3.8， CUDA 11.3

- 关于GPU服务器租用这里就不赘述了，详见我的另一篇博客：[常用深度学习平台的使用指南](https://www.eula.club/blogs/常用深度学习平台的使用指南.html)

### 2.1 大模型基座选型

选用当下效果比较好的Baichuan2-13B-Chat大模型，以下将会提供普通服务和流式服务两种调用方式。

- 项目地址：[https://github.com/baichuan-inc/Baichuan2](https://github.com/baichuan-inc/Baichuan2)

显存要求如下表所示，由于租用的3090显卡只有24GB显存，因此只能跑8bits量化模型。如果你的显卡资源够，可以跑全精度，代码改成model = model.cuda()

| Precision   | Baichuan2-7B | Baichuan2-13B |
| ----------- | ------------ | ------------- |
| bf16 / fp16 | 15.3         | 27.5          |
| 8bits       | 8.0          | 16.1          |
| 4bits       | 5.1          | 8.6           |

### 2.2 准备部署代码

这里未使用vLLM等技术对推理服务的性能进行优化，实际投入使用的时候建议开启vLLM，这样可以充分利用显卡计算资源，带来更好的推理性能，详见我的另一篇博客：[基于vLLM加速大模型推理服务](https://www.eula.club/blogs/基于vLLM加速大模型推理服务.html)

#### 2.2.1 普通服务的代码

baichuan_api_server.py

```python
# -*- coding: utf-8 -*-

from flask import Flask, request
from flask_cors import cross_origin
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import datetime

model_path = '/Path/Baichuan2-13B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                             trust_remote_code=True)
torch.cuda.set_device(0)  # 指定显卡
# model = model.cuda()
model = model.quantize(8).cuda()
model.generation_config = GenerationConfig.from_pretrained(
    model_path
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True
)
model.eval()

app = Flask(__name__)


@app.route('/', methods=['POST'])
@cross_origin()
def batch_chat():
    global model, tokenizer

    data = json.loads(request.get_data())
    now = datetime.datetime.now()
    time_format = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        messages = data.get("messages")
        response = model.chat(tokenizer, messages)
        answer = {"response": response, "history": [], "status": 200, "time": time_format}
        return answer
    except Exception as e:
        return {"response": f"大模型预测出错:{repr(e)}", "history": [('', '')], "status": 444, "time": time_format}


if __name__ == '__main__':
    with torch.no_grad():
        app.run(host='0.0.0.0', port=1707)
```

后台启动服务：

```
$ nohup python3 baichuan_api_server.py > baichuan_api_server.log 2>&1 &           
```

#### 2.2.2 流式服务的代码

baichuan_stream_api_server.py

```python
# -*- coding: utf-8 -*-

import argparse
from flask import Flask, request, Response
from flask_cors import cross_origin
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = '/Path/Baichuan2-13B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",
                                             trust_remote_code=True)
torch.cuda.set_device(0)  # 指定显卡
# model = model.cuda()
model = model.quantize(8).cuda()
model.generation_config = GenerationConfig.from_pretrained(
    model_path
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True
)
model.eval()

app = Flask(__name__)


def solve(messages):
    position = 0
    for response in model.chat(tokenizer, messages, stream=True):
        chunk = response[position:]
        yield chunk
        position = len(response)


@app.route('/', methods=['POST'])
@cross_origin()
def batch_chat():
    global model, tokenizer

    data = json.loads(request.get_data())
    messages = data.get("messages")
    return Response(solve(messages), content_type='text/plain; charset=utf-8')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--port', default=1708, type=int, help='服务端口')
args = parser.parse_args()

if __name__ == '__main__':
    with torch.no_grad():
        app.run(host='0.0.0.0', port=args.port)
```

后台启动服务：

```
$ nohup python3 baichuan_stream_api_server.py > baichuan_stream_api_server.log 2>&1 & 
```

### 2.3 下载模型并安装依赖

#### 2.3.1 下载模型文件

模型地址：[https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/tree/main](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/tree/main)

![Baichuan2-13B-Chat模型](README.assets/Baichuan2-13B-Chat模型.png)

注：如果没有梯子，也可以用国内镜像站去下载模型，[https://aifasthub.com/models](https://aifasthub.com/models)

可以使用 HuggingFace Hub 下载模型文件，首先，我们需要安装huggingface_hub依赖。

```
$ pip3 install huggingface_hub
```

之后执行该脚本即可。

```python
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
```

#### 2.3.2 安装依赖环境

torch环境使用服务器镜像自带的（没有的话 pip3 install torch 安装一下）。依赖安装的坑比较多，主要是CUDA环境不匹配的问题。

```
$ pip3 install flask 
$ pip3 install flask_cors
$ pip3 install accelerate 
$ pip3 install sentencepiece
$ pip3 install scipy
$ pip3 install transformers==4.33.2  
$ pip3 install xformers

$ git clone https://github.com/TimDettmers/bitsandbytes.git
$ cd bitsandbytes
$ vim Makefile
# CC_ADA_HOPPER := -gencode arch=compute_89,code=sm_89
# CC_ADA_HOPPER += -gencode arch=compute_90,code=sm_90
$ CUDA_VERSION=121 make cuda12x
$ python3 setup.py install
```

踩过的坑：

[1] transformers 安装问题

一开始直接使用 pip3 install transformers  去安装，但出现了 AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model' 的问题。检查模型文件下载全了，查阅资料得知 pip3 install transformers==4.33.2 版本可解决此问题。

安装完之后，执行时又卡在 Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers，之后又执行 pip3 install xformers 安装该依赖，解决了该问题。

[2] bitsandbytes安装问题

bitsandbytes是用于大模型量化的库，项目地址：[https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

一开始直接使用 pip3 install bitsandbytes 去安装，但出现了与CUDA不兼容的问题。（该问题不一定会出现，优先使用pip3去安装，不行的话再考虑编译安装）

![bitsandbytes与CUDA版本不兼容问题](README.assets/bitsandbytes与CUDA版本不兼容问题.png)

然后我又使用了编译安装的方式，又出现 Unsupported gpu architecture 'compute_89' 的问题。

![bitsandbytes编译安装的版本问题](README.assets/bitsandbytes编译安装的版本问题.png)

使用  nvcc --list-gpu-arch 命令查询，发现只支持到 compute_86。因此修改 Makefile，将compute_89和compute_90的都注释掉，然后重新进行编译即可。

### 2.4 使用大模型服务

#### 2.4.1 使用普通服务

baichuan_api_test.py

```python
# -*- coding: utf-8 -*-


import requests
import json


class Baichuan:
    def __init__(self, url):
        self.url = url

    def __call__(self, messages: list) -> str:
        data = {"messages": messages}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        return response["response"]


if __name__ == '__main__':
    llm = Baichuan("http://127.0.0.1:1707/")
    messages = [{
        "role": "user",
        "content": "解释一下量子计算"
    }]
    response = llm(messages)
    print(response)
```

#### 2.4.2 使用流式服务

baichuan_stream_api_test.py

```python
# -*- coding: utf-8 -*-

import requests
import json


class Baichuan:
    def __init__(self, url):
        self.url = url

    def stream_request(self, messages: list):
        data = json.dumps({"messages": messages})
        try:
            with requests.post(self.url, data=data, headers={'Content-Type': 'application/json'}, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_chunk = line.decode('utf-8')
                        yield decoded_chunk
        except requests.RequestException as e:
            print(f"请求错误: {e}")


if __name__ == '__main__':
    llm = Baichuan("http://127.0.0.1:1708")
    messages = [{
        "role": "user",
        "content": "解释一下量子计算"
    }]
    for response in llm.stream_request(messages):
        print(response)
```

注：使用流式服务，可以让结果随着推理过程，一点儿一点儿的往外输出，用户体验更好，但使用流式服务会比普通服务更耗资源。

#### 2.4.3 运行出的效果

以下是 Baichuan2-13B-Chat 模型在 8bits 量化的运行效果。

![Baichuan2-13B-Chat-8bits量化的运行效果](README.assets/Baichuan2-13B-Chat-8bits量化的运行效果.png)

#### 2.4.4 服务压力测试

可使用如下脚本对普通服务进行压测。

```python
# -*- coding: utf-8 -*-

from typing import Union, Any, List, Tuple
import requests
import json
import threading
import time


class Baichuan:
    def __init__(self, url):
        self.url = url

    def send_request(self, messages: List[dict]) -> Tuple[bool, Union[str, Any], float]:
        start_time = time.time()
        try:
            data = {"messages": messages}
            response = requests.post(self.url, json=data)
            response = json.loads(response.content)
            success = True
        except Exception as e:
            response = str(e)
            success = False
        end_time = time.time()
        return success, response, end_time - start_time


def worker(url, messages, index, stats):
    bc = Baichuan(url)
    success, response, duration = bc.send_request(messages)
    with stats['lock']:
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1

        stats['total_duration'] += duration
        if duration < stats['min_duration']:
            stats['min_duration'] = duration
        if duration > stats['max_duration']:
            stats['max_duration'] = duration

    print(f"Thread {index}: {'Success' if success else 'Failure'}, Response: {response}, Duration: {duration}s")


if __name__ == '__main__':
    url = "http://127.0.0.1:1707/"
    messages = [{
        "role": "user",
        "content": "解释一下量子计算"
    }]

    num_threads = 10  # 测试并发线程数
    num_rounds = 3  # 测试轮数

    stats = {
        'success_count': 0,
        'failure_count': 0,
        'total_duration': 0.0,
        'min_duration': float('inf'),
        'max_duration': float('-inf'),
        'lock': threading.Lock()
    }

    for round in range(num_rounds):
        print(f"开始测试轮数: {round + 1}")
        threads = []

        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(url, messages, i, stats))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    # 输出总体统计结果
    avg_duration = stats['total_duration'] / (num_threads * num_rounds) if num_threads > 0 else 0
    print(f"总成功次数: {stats['success_count']}, 总失败次数: {stats['failure_count']}")
    print(f"整体最短耗时: {stats['min_duration']:.2f}s, 整体最长耗时: {stats['max_duration']:.2f}s, 整体平均耗时: {avg_duration:.2f}s")
```

## 3. 检索增强大模型生成实例

实例场景概述：有一批内部的政府政策文档，数据不可外传，使用自部署的大模型来实现，需要基于这些文档进行垂直领域的问答。

本项目我已经在Github上进行了开源，项目地址为：[https://github.com/Logistic98/rag-llm](https://github.com/Logistic98/rag-llm)

数据不便于公开，所以用于构建检索库的数据文件只留了一个示例，用于构建检索库的数据这里是处理成JSON格式了。

![数据预处理后的格式-用于构建检索库](README.assets/数据预处理后的格式-用于构建检索库.png)

### 3.1 原始数据预处理

#### 3.1.1 数据预处理要求

数据预处理：需要将数据预处理成结构化数据之后，才能方便的构建检索库。

- 数据预处理要求：每个文档拆开，拆开后每个数据是文档中的某一段，目的是保证每条数据都有较完整的语义，并且长度不会太长。
- 数据预处理方式：提供的文档主要是Word、PDF等格式，无法直接使用。数据量少的话，可以直接人工去处理。数据量大的话，建议先使用脚本批量处理一下，有些解析不成功的再人工处理。

#### 3.1.2 数据预处理脚本

PDF格式是非常难处理的，如果是文本类型的可以使用以下脚本来初步处理，如果本身就是图片类型的，那该脚本解析不了，就需要OCR技术来辅助了。关于复杂PDF文件的解析可以使用 Marker 库，详见我的另一篇博客：[PDF解析与基于LLM的本地知识库问答](https://www.eula.club/blogs/PDF解析与基于LLM的本地知识库问答.html)

pdf2word.py

```python
# -*- coding: utf-8 -*-

import os
from pdf2docx import Converter
import argparse

parser = argparse.ArgumentParser(description="服务调用方法：python pdf2word.py --pdf_path 'xxx.pdf' --docx_path 'xxx.docx'")
parser.add_argument("--pdf_path", type=str, required=True, help="要解析的 PDF 文件地址")
parser.add_argument("--docx_path", type=str, required=True, help="解析后的 DOCX 文件输出地址")
args = parser.parse_args()

# 确保输出文件的目录存在
docx_dir = os.path.dirname(args.docx_path)
if not os.path.exists(docx_dir):
    os.makedirs(docx_dir)

try:
    # 初始化转换器并转换 PDF 到 DOCX
    cv = Converter(args.pdf_path)
    cv.convert(args.docx_path)  # 默认转换所有页面
    cv.close()
    print("PDF 文件已成功转换为 DOCX 格式。")
except Exception as e:
    print(f"转换过程中发生错误：{str(e)}")
```

word2json.py

```python
# -*- coding: utf-8 -*-

import os
from docx import Document
import json
import argparse

parser = argparse.ArgumentParser(description="服务调用方法：python word2json.py --docx_path 'xxx.docx' --output_path 'xxx.json' --max_length 500")
parser.add_argument("--docx_path", type=str, required=True, help="docx 文件地址")
parser.add_argument("--output_path", type=str, required=True, help="结果输出地址")
parser.add_argument("--max_length", default=500, type=int, help="切片大小")
args = parser.parse_args()

# 读取 DOCX 文件
docx = Document(args.docx_path)
max_length = args.max_length

result = []
current_text = ""
num_toolong = 0

for paragraph in docx.paragraphs:
    section = paragraph.text.strip()
    # 如果当前段落加上新段落的长度小于等于最大长度，或者当前文本为空
    if not current_text or len(current_text) + len(section) + 1 <= max_length:
        current_text += " " + section  # 添加空格作为分隔
    else:
        # 否则，将当前文本作为一个段落添加到结果中，并重新开始新的段落
        result.append(current_text.strip())
        if len(current_text) > max_length:
            num_toolong += 1
        current_text = section

# 添加最后一段文字
if current_text:
    result.append(current_text.strip())

# 检查输出目录是否存在，如果不存在，则创建
output_dir = os.path.dirname(args.output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将结果写入 JSON 文件
with open(args.output_path, "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)

print("finish")
```

### 3.2 构建ES检索库

#### 3.2.1 搭建ES环境并安装ik分词器

Step1：搭建Docker环境

```
$ apt-get update -y && apt-get install curl -y  # 安装curl
$ curl https://get.docker.com | sh -   # 安装docker
$ sudo systemctl start docker  # 启动docker服务（改成restart即为重启服务）
$ docker version # 查看docker版本（客户端要与服务端一致）
```

Step2：使用Docker搭建ElasticSearch

```
$ docker pull elasticsearch:7.16.2
$ docker run -d --name es \
-p 9200:9200 -p 9300:9300 \
-e "discovery.type=single-node" -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
elasticsearch:7.16.2
$ docker update es --restart=always
```

Step3：进入容器给ElasticSearch配置密码

```
$ docker exec -it es /bin/bash 
$ cd config
$ chmod o+w elasticsearch.yml
$ vi elasticsearch.yml
```

其中，在 elasticsearch.yml 文件的末尾添加以下配置，代表开启xpack安全认证）

```
xpack.security.enabled: true    
```

然后把权限修改回来，重启容器，设置账号密码，浏览器访问`http://IP:9200`地址即可（用 elastic账号 和自己设置的密码登录即可）

```
$ chmod o-w elasticsearch.yml
$ exit
$ docker restart es
$ docker exec -it es /bin/bash 
$ ./bin/elasticsearch-setup-passwords interactive   // 然后设置一大堆账号密码
```

Step4：安装ik分词器插件

```
$ docker exec -it es /bin/bash
$ apt-get install -y wget   
$ wget https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v7.16.2/elasticsearch-analysis-ik-7.16.2.zip
$ unzip -o -d /usr/share/elasticsearch/elasticsearch-analysis-ik-7.16.2 /usr/share/elasticsearch/elasticsearch-analysis-ik-7.16.2.zip
$ rm –f elasticsearch-analysis-ik-7.16.2.zip
$ mv /usr/share/elasticsearch/elasticsearch-analysis-ik-7.16.2 /usr/share/elasticsearch/plugins/ik
$ cd /usr/share/elasticsearch/bin
$ elasticsearch-plugin list
$ exit
$ docker restart es
```

#### 3.2.2 构建ES索引并写入数据

安装 elasticsearch 依赖

```
$ pip3 install elasticsearch
```

es_index.py

```python
# -*- coding: utf-8 -*-

import json
from elasticsearch import Elasticsearch
from elasticsearch import helpers

index_name = "policy_qa"
es = Elasticsearch(
    hosts=["http://127.0.0.1:9200"],
    basic_auth=("elastic", "your_password"),
    request_timeout=60
)
CREATE_BODY = {
    "settings": {
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "ik_max_word"
            }
        }
    }
}

es.indices.create(index=index_name, body=CREATE_BODY)
contents = []

with open("./preprocess_data/preprocess_data.json", "r", encoding="utf-8") as file:
    temp = json.load(file)
contents = contents + temp

action = (
    {
        "_index": index_name,
        "_type": "_doc",
        "_id": i,
        "_source": {
            "content": contents[i]
        }
    } for i in range(0, len(contents))
)
helpers.bulk(es, action)

print("export es finish")
```

执行该文件，将预处理的数据导入ES索引库。

![将预处理的数据导入ES索引库](README.assets/将预处理的数据导入ES索引库.png)

#### 3.2.3 构建ES文档检索服务

es_search.py

```python
# -*- coding: utf-8 -*-

import json
from flask import Flask, request
from flask_cors import cross_origin
from elasticsearch import Elasticsearch

index_name = "policy_qa"
es = Elasticsearch(
    hosts=["http://127.0.0.1:9200"],
    basic_auth=("elastic", "your_password"),
    request_timeout=60
)

app = Flask(__name__)


@app.route('/', methods=['POST'])
@cross_origin()
def retrieval():
    data = json.loads(request.get_data())
    question = data.get("question")
    top_k = data.get("top_k")
    query_body = {
        "query": {
            "match": {
                "content": question
            }
        },
        "size": top_k
    }
    res = es.search(index=index_name, body=query_body)
    docs = []
    for hit in res['hits']['hits']:
        docs.append(hit["_source"]["content"])
    return {"docs": docs}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1709)
```

启动ES检索服务，下面会用到。

### 3.3 基于ES检索增强生成回答

solve.py

```python
# -*- coding: utf-8 -*-

import os
import requests
import json

# Global Parameters
RETRIEVAL_TOP_K = 2
LLM_HISTORY_LEN = 30


class Baichuan:
    def __init__(self, url):
        self.url = url

    def __call__(self, messages: list) -> str:
        data = {"messages": messages}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        return response["response"]


def init_cfg(url_llm):
    global llm
    llm = Baichuan(url=url_llm)


def get_docs(question: str, url: str, top_k=RETRIEVAL_TOP_K):
    data = {"question": question, "top_k": top_k}
    docs = requests.post(url, json=data)
    docs = json.loads(docs.content)
    return docs["docs"]


def get_knowledge_based_answer(query, history_obj, url_retrieval):
    global llm, RETRIEVAL_TOP_K

    if len(history_obj.history) > LLM_HISTORY_LEN:
        history_obj.history = history_obj.history[-LLM_HISTORY_LEN:]

    # Rewrite question
    if len(history_obj.history):
        rewrite_question_input = history_obj.history.copy()
        rewrite_question_input.append(
            {
                "role": "user",
                "content": f"""请基于对话历史，对后续问题进行补全重构，如果后续问题与历史相关，你必须结合语境将代词替换为相应的指代内容，让它的提问更加明确；否则直接返回原始的后续问题。
                注意：请不要对后续问题做任何回答和解释。
                
                后续问题：{query}
                
                修改后的后续问题："""
            }
        )
        new_query = llm(rewrite_question_input)
    else:
        new_query = query

    # 获取相关文档
    docs = get_docs(new_query, url_retrieval, RETRIEVAL_TOP_K)
    doc_string = ""
    for i, doc in enumerate(docs):
        doc_string = doc_string + doc + "\n"
    history_obj.history.append(
        {
            "role": "user",
            "content": f"请基于参考，回答问题，并给出参考依据：\n问题：\n{query}\n参考：\n{doc_string}\n答案："
        }
    )

    # 调用大模型获取回复
    response = llm(history_obj.history)

    # 修改history，将之前的参考资料从history删除，避免history太长
    history_obj.history[-1] = {"role": "user", "content": query}
    history_obj.history.append({"role": "assistant", "content": response})

    # 检查history.json是否存在，如果不存在则创建
    if not os.path.exists("./history.json"):
        with open("./history.json", "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=2)

    # 读取现有数据，追加新数据，并写回文件
    with open("./history.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    data.append({"query": query, "new_query": new_query, "docs": docs, "response": response,
                 "retrieval": "ES"})
    with open("./history.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return {"response": response, "docs": docs}
```

server.py

```python
# -*- coding: utf-8 -*-

from flask import Flask, request
from flask_cors import cross_origin
import time
from solve import *

app = Flask(__name__)


class History:
    def __init__(self):
        self.history = []


session_histories = {}


@app.route("/get_bot_response", methods=["POST"])
@cross_origin()
def get_bot_response():
    global session_histories
    data = json.loads(request.get_data())
    userText = data["content"]  # 用户输入
    session_id = data["id"]  # 用户id，用于保存对话历史

    # 获取对话历史，如果有的话
    if session_id in session_histories:
        history_obj = session_histories[session_id]["history"]
        session_histories[session_id]["last_access_time"] = time.time()
    else:
        history_obj = History()
        session_histories[session_id] = {
            "history": history_obj,
            "last_access_time": time.time(),
        }

    # 如果用户超过一个小时没有交互，则删除该用户的对话历史
    max_idle_time = 60 * 60
    for session_id, session_data in session_histories.copy().items():
        idle_time = time.time() - session_data["last_access_time"]
        if idle_time > max_idle_time:
            del session_histories[session_id]

    if userText == "清空对话历史":
        history_obj.history = []
        return str("已清空")

    response = get_knowledge_based_answer(
        query=userText, history_obj=history_obj, url_retrieval="http://127.0.0.1:1709/"
    )
    return response


if __name__ == "__main__":
    init_cfg("http://127.0.0.1:1707/")
    app.run(host="0.0.0.0", port=1710)
```

rag_test.py

```python
# -*- coding: utf-8 -*-

import json
import requests
import random
from tqdm import trange


if __name__ == '__main__':
    url = "http://127.0.0.1:1710/get_bot_response"
    question = ["什么是政府专项债务？", "专项债收入可以用于经常性支出吗？", "政府专项债务应当通过什么偿还？"]
    for i in trange(len(question)):
        data = {"id": random.randint(0, 9999999), "content": question[i]}
        res = requests.post(url, json=data)
        res = json.loads(res.content)
        print("\nQuestion: " + question[i])
        print("\nAnswer: " + res["response"])
        print("\n-------------------------------------------------")
```

先执行 server.py 启动 ES 检索增强大模型生成服务，再执行 rag_test.py 进行测试。输出里会有个 history.json 文件，记录中间过程及结果。

![基于ES检索增强生成回答的效果](README.assets/基于ES检索增强生成回答的效果.png)