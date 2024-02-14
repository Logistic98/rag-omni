# -*- coding: utf-8 -*-

import argparse
from flask import Flask, request, Response
from flask_cors import cross_origin
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = 'baichuan-inc/Baichuan2-13B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",
                                             trust_remote_code=True)
torch.cuda.set_device(0)  # 指定显卡
model = model.cuda()
# model = model.quantize(8).cuda()
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
