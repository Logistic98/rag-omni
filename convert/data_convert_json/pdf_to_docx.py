# -*- coding: utf-8 -*-

import os
from pdf2docx import Converter
import argparse

parser = argparse.ArgumentParser(description="服务调用方法：python3 pdf_to_docx.py --pdf_path 'xxx.pdf' --docx_path 'xxx.docx'")
parser.add_argument("--pdf_path", type=str, required=True, help="要解析的 PDF 文件地址")
parser.add_argument("--docx_path", type=str, required=True, help="解析后的 DOCX 文件输出地址")
args = parser.parse_args()

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
