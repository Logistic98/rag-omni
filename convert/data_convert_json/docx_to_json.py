# -*- coding: utf-8 -*-

import os
from docx import Document
import json
import argparse

parser = argparse.ArgumentParser(description="服务调用方法：python3 docx_to_json.py --docx_path 'xxx.docx' --output_path 'xxx.json' --max_length 500")
parser.add_argument("--docx_path", type=str, required=True, help="docx 文件地址")
parser.add_argument("--output_path", type=str, required=True, help="结果输出地址")
parser.add_argument("--max_length", default=500, type=int, help="切片大小")
args = parser.parse_args()

docx = Document(args.docx_path)
max_length = args.max_length

result = []
current_text = ""

for paragraph in docx.paragraphs:
    section = paragraph.text.strip()
    if not current_text or len(current_text) + len(section) + 1 <= max_length:
        current_text += " " + section
    else:
        result.append({
            "file_name": os.path.basename(args.docx_path),
            "part_content": current_text.strip()
        })
        current_text = section

if current_text:
    result.append({
        "file_name": os.path.basename(args.docx_path),
        "part_content": current_text.strip()
    })

output_dir = os.path.dirname(args.output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(args.output_path, "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)

print(f"{args.docx_path} 处理完成")
