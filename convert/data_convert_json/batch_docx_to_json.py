# -*- coding: utf-8 -*-

import os
import subprocess

if __name__ == '__main__':

    input_dir = "../../data/original_data"  # docx 文件目录
    output_dir = "../../data/preprocess_data_temp"  # json 结果输出目录
    max_length = 500  # 切片大小

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".docx"):
            docx_path = os.path.join(input_dir, filename)
            output_filename = filename.replace(".docx", ".json")
            output_path = os.path.join(output_dir, output_filename)
            cmd = [
                "python3", "docx_to_json.py",
                "--docx_path", docx_path,
                "--output_path", output_path,
                "--max_length", str(max_length)
            ]
            subprocess.run(cmd)

    print("所有 docx 文件已成功转换为 json 文件。")
