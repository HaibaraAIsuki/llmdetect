# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:47:56 2025

@author: ruiha
"""

import sys
import subprocess

def main():
    if len(sys.argv) < 3:
        print("Usage: python dispatch.py /path/to/image_or_folder model_name")
        sys.exit(1)

    input_path = sys.argv[1]
    model_name = sys.argv[2]

    # 判断模型是否为 Ollama
    if model_name == "llama3.2-vision" :
        # 如果是 Ollama，调用当前脚本的处理函数
        print("Using Ollama model...")
        subprocess.run(["python", "detect_tag.py", input_path, model_name])
    else:
        # 如果是其他模型，调用未来的脚本（假设是其他脚本）
        print(f"Using {model_name} model...")
        subprocess.run(["python", f"{model_name}_inference.py", input_path, model_name])

if __name__ == "__main__":
    main()
