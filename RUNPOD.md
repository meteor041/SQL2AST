# RunPod 运行备忘录

## 环境保存方式

RunPod 里要区分两类东西：

- `/workspace`：Volume/Network Volume 挂载目录。数据、模型、代码、venv 放这里，下次挂同一个 volume 还能用。
- 系统目录：例如 `/usr`, `/opt`, `/root`。普通 Pod 删除或重建后不保证保留，`apt-get install openjdk` 这类系统安装需要重做，除非做成自定义 Docker 镜像。

短期推荐：

```bash
/workspace/venvs/sqlrm
/workspace/emb/lxy/sql_rm
/workspace/emb/lxy/csc_sql
/workspace/data/ches
/workspace/models/Qwen3-4B-Instruct-2507
/workspace/hf_cache
/workspace/tmp
```

这样下次新建 Pod 并挂载同一个 Network Volume 后，只需要补系统包和激活 venv：

```bash
apt-get update
apt-get install -y openjdk-21-jdk rsync rclone

source /workspace/venvs/sqlrm/bin/activate
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH=$JAVA_HOME/bin:$PATH
```

长期推荐：做一个自定义 Docker 镜像，把 Python 包、vLLM、OpenJDK、rclone 都预装进去。这样新建 Pod 后只需要挂载 `/workspace`，不用重新装依赖。

## 自定义 Docker 镜像

仓库里提供了：

```text
docker/Dockerfile.runpod
docker/requirements-runpod.txt
```

镜像里预装：

```text
CUDA 12.8
Python 3.11
OpenJDK 17
git / rsync / rclone / awscli
vLLM / torch 依赖
transformers / peft / trl / datasets / accelerate
pyserini / pyjnius / nltk / modelscope
```

模型、数据、代码和结果仍然放 Network Volume 的 `/workspace`，不要放进镜像。

构建镜像：

```bash
cd /home/pkuccadm/huwenp/emb/lxy/sql_rm

docker build \
  -f docker/Dockerfile.runpod \
  -t your_dockerhub_user/sqlrm-runpod:cuda128-vllm \
  .
```

推到 Docker Hub：

```bash
docker login
docker push your_dockerhub_user/sqlrm-runpod:cuda128-vllm
```

或者推到 GHCR：

```bash
docker login ghcr.io
docker tag your_dockerhub_user/sqlrm-runpod:cuda128-vllm ghcr.io/your_github_user/sqlrm-runpod:cuda128-vllm
docker push ghcr.io/your_github_user/sqlrm-runpod:cuda128-vllm
```

RunPod 新建 Pod 时：

```text
Container image: your_dockerhub_user/sqlrm-runpod:cuda128-vllm
Volume: lxy-volume
Mount path: /workspace
```

启动后只需要恢复代码/数据路径并设置环境变量：

```bash
cd /workspace
mkdir -p /workspace/emb/lxy /workspace/tmp /workspace/hf_cache /data/model

test -d /workspace/emb/lxy/sql_rm || git clone https://github.com/meteor041/SQL2AST.git /workspace/emb/lxy/sql_rm
test -d /workspace/emb/lxy/csc_sql || git clone https://github.com/meteor041/csc-sql.git /workspace/emb/lxy/csc_sql

ln -sfn /workspace/models/Qwen3-4B-Instruct-2507 /data/model/Qwen3-4B-Instruct-2507

export PROJECT_ROOT=/workspace/emb/lxy/sql_rm
export CSC_SQL_ROOT=/workspace/emb/lxy/csc_sql
export DATA_ROOT=/workspace/data/ches
export CUDA_VISIBLE_DEVICES=0
export TENSOR_PARALLEL_SIZE=1
export TMPDIR=/workspace/tmp
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH=$JAVA_HOME/bin:$PATH
export VLLM_USE_MODELSCOPE=False
export N_SQL_GENERATE=8
export WAIT_FOR_GPU_IDLE=false

cd /workspace/emb/lxy/sql_rm
```

检查镜像环境：

```bash
python - <<'PY'
import torch, vllm, transformers, peft, trl, datasets, accelerate
import nltk, pyserini
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("env ok")
PY
```

## 初始化 Network Volume 环境

Network Volume 只能保存文件，不能离线执行安装。第一次仍然要启动一个 Pod，并把 `lxy-volume` 挂载到 `/workspace`，然后把 Python venv、代码、缓存写到 `/workspace`。之后新 Pod 挂同一个 volume 就能复用。

在新 Pod 里执行：

```bash
cd /workspace

apt-get update
apt-get install -y openjdk-21-jdk rsync rclone git

mkdir -p /workspace/venvs /workspace/tmp /workspace/pip_cache /workspace/hf_cache
python3 -m venv /workspace/venvs/sqlrm
source /workspace/venvs/sqlrm/bin/activate

export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip_cache
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache

pip install -U pip setuptools wheel
pip install --no-cache-dir vllm
```

克隆代码：

```bash
mkdir -p /workspace/emb/lxy
cd /workspace/emb/lxy

git clone https://github.com/meteor041/SQL2AST.git sql_rm
git clone https://github.com/meteor041/csc-sql.git csc_sql
```

安装项目依赖：

```bash
cd /workspace/emb/lxy/sql_rm

grep -v -E '^(torch|torchvision|torchaudio|nvidia-|triton|vllm)' requirements.txt > /tmp/requirements-no-heavy.txt
pip install --no-cache-dir -r /tmp/requirements-no-heavy.txt

pip install --no-cache-dir nltk==3.8.1 ijson func-timeout pyserini pyjnius modelscope
python - <<'PY'
import nltk
nltk.download("punkt")
PY
```

安装 `csc_sql`：

```bash
cd /workspace/emb/lxy/csc_sql
pip install --no-cache-dir -e . --no-deps
```

写入下次一键恢复脚本：

```bash
cat > /workspace/setup_env.sh <<'EOF'
#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y openjdk-21-jdk rsync rclone git

source /workspace/venvs/sqlrm/bin/activate

export PROJECT_ROOT=/workspace/emb/lxy/sql_rm
export CSC_SQL_ROOT=/workspace/emb/lxy/csc_sql
export DATA_ROOT=/workspace/data/ches
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export TMPDIR=/workspace/tmp
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH=$JAVA_HOME/bin:$PATH
export VLLM_USE_MODELSCOPE=False
export N_SQL_GENERATE="${N_SQL_GENERATE:-8}"
export WAIT_FOR_GPU_IDLE=false

mkdir -p /workspace/tmp /workspace/hf_cache
mkdir -p /data/model
ln -sfn /workspace/models/Qwen3-4B-Instruct-2507 /data/model/Qwen3-4B-Instruct-2507

cd /workspace/emb/lxy/sql_rm

echo "Environment ready."
echo "Run eval:"
echo "  WAIT_FOR_GPU_IDLE=false bash scripts/04_eval_sft.sh"
echo "  WAIT_FOR_GPU_IDLE=false bash scripts/05_eval_dpo.sh"
EOF

chmod +x /workspace/setup_env.sh
```

检查环境：

```bash
source /workspace/setup_env.sh

python - <<'PY'
import torch, vllm, transformers, peft, trl, datasets, accelerate, cscsql
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("env ok")
PY
```

下次新建 Pod 并挂载 `lxy-volume` 后，只需要：

```bash
source /workspace/setup_env.sh
```

## 从 Cloudflare R2 恢复数据

新建 RunPod 后，先把数据和模型从 R2 拉到 `/workspace`，避免从本服务器直接传输时长时间占用 GPU。

```bash
mkdir -p /workspace/data/ches
mkdir -p /workspace/models

rclone copy r2:ches/dev /workspace/data/ches -P
rclone copy r2:ches/outputs /workspace/data/ches/outputs -P
rclone copy r2:ches/models/Qwen3-4B-Instruct-2507 /workspace/models/Qwen3-4B-Instruct-2507 -P
```

## 预填充 Network Volume

已创建的 Network Volume：

```text
Name: lxy-volume
Datacenter: US-CA-2
S3 endpoint: https://s3api-us-ca-2.runpod.io/
```

RunPod S3 API 的 bucket 通常是 Network Volume ID，不一定是显示名 `lxy-volume`。在 RunPod Storage 页面复制这个 volume 的 ID，下面用 `RUNPOD_VOLUME_ID` 表示。

在本服务器或任意不占 GPU 的机器上配置 RunPod Network Volume 的 rclone remote：

```bash
rclone config
```

推荐配置：

```text
name> runpodnv
Storage> s3
provider> Other
access_key_id> 你的 RunPod S3 Access Key
secret_access_key> 你的 RunPod S3 Secret Key
endpoint> https://s3api-us-ca-2.runpod.io/
region> US-CA-2
```

从 Cloudflare R2 预填充到 RunPod Network Volume：

```bash
export RUNPOD_VOLUME_ID=你的_network_volume_id

rclone copy r2:ches/dev \
  runpodnv:${RUNPOD_VOLUME_ID}/data/ches \
  -P

rclone copy r2:ches/outputs \
  runpodnv:${RUNPOD_VOLUME_ID}/data/ches/outputs \
  -P

rclone copy r2:ches/models/Qwen3-4B-Instruct-2507 \
  runpodnv:${RUNPOD_VOLUME_ID}/models/Qwen3-4B-Instruct-2507 \
  -P
```

之后部署 GPU Pod 时选择 `lxy-volume`，mount path 保持 `/workspace`。Pod 启动后应能直接看到：

```bash
ls -lh /workspace/data/ches
ls -lh /workspace/data/ches/outputs
ls -lh /workspace/models/Qwen3-4B-Instruct-2507
```

如果 LoRA 的 `adapter_config.json` 里仍然指向原服务器路径 `/data/model/Qwen3-4B-Instruct-2507`，在 RunPod 上创建软链接：

```bash
mkdir -p /data/model
ln -sfn /workspace/models/Qwen3-4B-Instruct-2507 /data/model/Qwen3-4B-Instruct-2507
```

## Eval 环境变量

```bash
cd /workspace/emb/lxy/sql_rm
source /workspace/venvs/sqlrm/bin/activate

export PROJECT_ROOT=/workspace/emb/lxy/sql_rm
export CSC_SQL_ROOT=/workspace/emb/lxy/csc_sql
export DATA_ROOT=/workspace/data/ches
export CUDA_VISIBLE_DEVICES=0
export TENSOR_PARALLEL_SIZE=1
export TMPDIR=/workspace/tmp
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH=$JAVA_HOME/bin:$PATH
export N_SQL_GENERATE=8
```

## 运行 Eval

```bash
WAIT_FOR_GPU_IDLE=false bash scripts/04_eval_sft.sh
WAIT_FOR_GPU_IDLE=false bash scripts/05_eval_dpo.sh
```

日志和输出在 `csc_sql` 仓库：

```bash
ls -lt /workspace/emb/lxy/csc_sql/logs | head
ls -lt /workspace/emb/lxy/csc_sql/outputs | head
```

## 重新训练

重新训练需要训练集、数据库和候选 SQL/DPO 数据。建议先从 R2 恢复到 `/workspace/data/ches`：

```bash
mkdir -p /workspace/data/ches

rclone copy r2:ches/train.json /workspace/data/ches/train.json -P
rclone copy r2:ches/train_databases /workspace/data/ches/train_databases -P
rclone copy r2:ches/candidates /workspace/data/ches/candidates -P
rclone copy r2:ches/data /workspace/data/ches/data -P
```

如果 R2 上没有 `data/dpo_pairs.jsonl`，先在本仓库生成 DPO 数据：

```bash
cd /workspace/emb/lxy/sql_rm
source /workspace/venvs/sqlrm/bin/activate

export PROJECT_ROOT=/workspace/emb/lxy/sql_rm
export DATA_ROOT=/workspace/data/ches
export CUDA_VISIBLE_DEVICES=0
export NUM_CPUS=8

bash scripts/01_prepare_and_process.sh
```

### RTX 4090 推荐训练配置

一张 RTX 4090 24GB 适合跑 Qwen3-4B 的 LoRA 训练，不适合全参训练。推荐先用下面配置跑通：

| 阶段 | 推荐配置 | 说明 |
| --- | --- | --- |
| SFT | batch 1, grad acc 16, seq 1024 | 稳，24GB 通常够 |
| DPO | batch 1, grad acc 16, seq 1024, prompt 768 | DPO 更吃显存，先保守 |
| Eval | tensor parallel 1 | 单卡推理即可 |

如果 SFT 显存还有余量，可以把 `SFT_MAX_SEQ_LENGTH` 提到 `1536` 或 `2048`。如果 DPO OOM，优先把 `DPO_MAX_SEQ_LENGTH` 降到 `768`，再把 `DPO_MAX_PROMPT_LENGTH` 降到 `512`。

单张 RTX 4090 训练 SFT：

```bash
cd /workspace/emb/lxy/sql_rm
source /workspace/venvs/sqlrm/bin/activate

export PROJECT_ROOT=/workspace/emb/lxy/sql_rm
export DATA_ROOT=/workspace/data/ches
export MODEL_NAME_OR_PATH=/workspace/models/Qwen3-4B-Instruct-2507
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export WAIT_FOR_GPU_IDLE=false
export SFT_PER_DEVICE_TRAIN_BATCH_SIZE=1
export SFT_GRADIENT_ACCUMULATION_STEPS=16
export SFT_MAX_SEQ_LENGTH=1024
export SFT_DATALOADER_NUM_WORKERS=0

bash scripts/02_train_sft.sh
```

单张 RTX 4090 训练 DPO。DPO 比 SFT 更吃显存，先用保守参数跑通：

```bash
cd /workspace/emb/lxy/sql_rm
source /workspace/venvs/sqlrm/bin/activate

export PROJECT_ROOT=/workspace/emb/lxy/sql_rm
export DATA_ROOT=/workspace/data/ches
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export WAIT_FOR_GPU_IDLE=false
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=1
export DPO_GRADIENT_ACCUMULATION_STEPS=16
export DPO_MAX_SEQ_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=768
export DPO_DATALOADER_NUM_WORKERS=0

bash scripts/03_train_dpo.sh
```

训练输出默认写到：

```bash
ls -lh /workspace/data/ches/outputs/sft
ls -lh /workspace/data/ches/outputs/dpo
```

训练完成后可以同步回 R2：

```bash
rclone copy /workspace/data/ches/outputs/sft r2:ches/outputs/sft -P
rclone copy /workspace/data/ches/outputs/dpo r2:ches/outputs/dpo -P
```
