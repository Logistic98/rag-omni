# -*- coding: utf-8 -*-

from flask import Flask, request
from flask_cors import cross_origin
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import datetime

model_path = 'baichuan-inc/Baichuan2-13B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
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
        return {"response": f"大模型预测出错:{repr(e)}", "history": [('', '')], "status": 400, "time": time_format}


if __name__ == '__main__':
    with torch.no_grad():
        app.run(host='0.0.0.0', port=1707)
