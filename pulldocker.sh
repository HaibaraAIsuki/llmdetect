#!/bin/bash

# 设置要拉取的模型列表
models=("llama3.2-vision" "llava-llama3" "llava:13b")

# 拉取 Ollama 的 Docker 镜像
docker pull docker.1ms.run/ollama/ollama

# 运行 Ollama 容器
docker run -d --name ollama_container -p 11434:11434 ollama/ollama

# 等待容器启动
echo "等待容器启动..."
sleep 10  # 根据需要调整等待时间

# 在容器中拉取指定的模型
for model in "${models[@]}"; do
    echo "正在拉取模型：$model"
    docker exec ollama_container ollama pull "$model"
done

echo "所有模型已成功拉取。"
