ln -s /home/ubuntu/MultiagentSystem /root/MultiagentSystem
cp -r /home/ubuntu/MultiagentSystem /mnt/shared/
mkdir -p ~/.docker/cli-plugins
curl -L --progress-bar https://github.com/docker/buildx/releases/download/v0.30.1/buildx-v0.30.1.linux-amd64 -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx

git fetch --all
git checkout ThunderReact

huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir /mnt/shared/Qwen3-Coder-30B-A3B-Instruct --local-dir-use-symlinks False
huggingface-cli download zai-org/GLM-4.6-FP8 --local-dir /mnt/shared/models/GLM-4.6-FP8
huggingface-cli download Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8  --local-dir /mnt/shared/models/Qwen3-Coder-480B-A35B-Instruct-FP8 
https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html


apt-get install -y python3.12-dev python3-dev build-essential
source .venv/bin/activate

vllm serve /scratch/models/GLM-4.6-FP8/GLM-4.6-FP8 \
     --tensor-parallel-size 8 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice


vllm bench serve \
  --model /mnt/shared/models/GLM-4.6-FP8 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos