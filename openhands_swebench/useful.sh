ln -s /home/ubuntu/MultiagentSystem /root/MultiagentSystem
cp -r /home/ubuntu/MultiagentSystem /mnt/shared/
mkdir -p ~/.docker/cli-plugins
curl -L --progress-bar https://github.com/docker/buildx/releases/download/v0.30.1/buildx-v0.30.1.linux-amd64 -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx

git fetch --all
git checkout ThunderReact

huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir /mnt/shared/Qwen3-Coder-30B-A3B-Instruct --local-dir-use-symlinks False

apt-get install -y python3.12-dev python3-dev build-essential
source .venv/bin/activate