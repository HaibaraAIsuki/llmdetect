import sys
import base64
import requests
from PIL import Image
from io import BytesIO
import re
import os
import json
import matplotlib.pyplot as plt

# 如果你的 Ollama 端口不同，请在此处修改
API_URL = "http://localhost:11435/api/generate"

# 全局变量：是否显示标注（可根据需要使用或去掉）
SHOW_ANNOTATION = True

def encode_image(image_path):
    """将图像转换为 base64 编码"""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_base64
    except Exception as e:
        print(f"Error encoding image: {e}")
        sys.exit(1)

def send_request(image_base64, prompt, model):
    """向 Ollama API 发送 POST 请求"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [image_base64]
    }
    try:
        response = requests.post(API_URL, json=payload)
        return response
    except Exception as e:
        print(f"Error sending request: {e}")
        sys.exit(1)

def extract_whitelisted_words(text, whitelist):
    """从文本中提取与 whitelist 匹配的词（忽略大小写），返回去重列表"""
    text_lower = text.lower()
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    
    matched = [word for word in whitelist if word.lower() in words_in_text]
    # 去重并保持顺序
    matched_unique = list(dict.fromkeys(matched))
    return matched_unique

def load_config():
    """从同级目录加载配置文件"""
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构造 config.json 的完整路径
    config_path = os.path.join(script_dir, "config.json")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config["question_list"]
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到。")
        sys.exit(1)
    except KeyError:
        print(f"配置文件中缺少 'question_list' 字段。")
        sys.exit(1)

def process_image(image_path, model, question_list):
    """
    对单张图片依次执行多次独立提问。
    - 每个问题都有自己的 whitelist，若无匹配，记为 "none"。
    - 最终返回类似:
      {
        "image_path": "xxx.jpg",
        "weather": "sunny",
        "position": "outdoors",
        "environment": "none",
        "obstacles": ...
      }
    """
    # 先准备输出 dict，并记录 image_path
    result_for_image = {
        "image_path": image_path,
        "weather": "none",
        "position": "none",
        "environment": "none",
        "obstacles": []
    }

    image_base64 = encode_image(image_path)

    # 对于 question_list 中的每个问题，独立调用 API
    for q_idx, item in enumerate(question_list, start=1):
        key_name = item["key"]        # 如 "weather"
        prompt = item["prompt"]       # 要问的问题
        whitelist = item["whitelist"] # 白名单

        # 调用 Ollama
        response = send_request(image_base64, prompt, model)
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            sys.exit(1)

        try:
            resp_json = response.json()
            response_text = resp_json.get("response", "").strip()
            # 提取本次问题对应的白名单关键词
            matched_words = extract_whitelisted_words(response_text, whitelist)

            # 如果找到多个匹配，这里示例只取第一个，也可自行改为拼接
            matched_value = matched_words[0] if matched_words else "none"

            # 将本次匹配结果写进相应字段
            result_for_image[key_name] = matched_value

            # 日志输出
            print(f"\n[Question {q_idx}] {key_name.upper()}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response_text}")
            print(f"Matched Words: {matched_words}")

        except Exception as e:
            print(f"Error parsing response: {e}")
            sys.exit(1)

    # 如果 position 字段为 "indoors"，则自动将 weather 字段清零
    if result_for_image.get("position", "").lower() in( "indoors","indoor"):
        result_for_image["weather"] = "null"

    return result_for_image

def process_images(image_path, model, question_list):
    """
    处理单个图像或文件夹中的所有图像，返回一个结果列表
    """
    all_results = []
    if os.path.isdir(image_path):
        for file_name in os.listdir(image_path):
            file_path = os.path.join(image_path, file_name)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_result = process_image(file_path, model, question_list)
                if img_result:
                    all_results.append(img_result)
    else:
        # 单张图像
        img_result = process_image(image_path, model, question_list)
        if img_result:
            all_results.append(img_result)

    return all_results

def run_inference(image_path, model):
    """
    执行推理，生成 output.json 文件并返回结果
    """
    # 加载配置文件（这里默认为 'config.json'，可根据需要修改）
    question_list = load_config('config.json')

    # 处理图像/文件夹，返回结果列表
    results = process_images(image_path, model, question_list)

    # 写入 output.json
    output_json = "output.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n处理完成！结果已写入 {output_json}。")

    return results

def load_json(json_file):
    """加载 JSON 数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def display_image_with_annotations(image_path, annotations):
    """显示图片和标记标签"""
    # 打开图片
    image = Image.open(image_path)

    # 使用 Matplotlib 显示图片
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴

    # 在图片上显示标注信息
    if SHOW_ANNOTATION:
        for annotation in annotations:
            label = (
                f"Weather: {annotation['weather']}, \n"
                f"Position: {annotation['position']}, \n"
                f"Environment: {annotation['environment']},\n "
                f"Obstacles: {annotation['obstacles']}\n"
            )
            position = [10, 10]  # 设置标签显示的位置（可根据需要调整）

            # 在图片上显示标签
            plt.text(
                position[0], position[1],
                label,
                color='red',
                fontsize=12,
                ha='left',
                va='top',
                bbox=dict(facecolor='white', alpha=0.5)
            )

    # 显示带标记标签的图片
    plt.show()
    plt.savefig("./pic", bbox_inches='tight', pad_inches=0.1)
    print("Image saved to .pic")

def display_results(json_file):
    """
    从 JSON 文件加载推理结果并可视化。
    """
    data = load_json(json_file)

    # 遍历每个图像及其标记
    for image_data in data:
        image_path = image_data["image_path"]
        annotations = [image_data]  # 每个图像的信息作为一个列表

        # 显示带标记标签的图片
        display_image_with_annotations(image_path, annotations)

def main():
    # 命令行参数：detect_tag.py /path/to/image_or_folder model_name
    if len(sys.argv) < 3:
        print("Usage: python detect_tag.py /path/to/image_or_folder model_name")
        sys.exit(1)

    input_path = sys.argv[1]
    model_name = sys.argv[2]

    # 执行推理并生成 output.json
    run_inference(input_path, model_name)

    # 加载并显示推理结果
    display_results("output.json")

if __name__ == "__main__":
    main()
