set -x

# Fully async GRPO training+generation for Qwen3-32B on SWE-Bench (SWE-Gym full dataset).
# Uses Megatron-LM as the training backend with PP=4 across 4 training nodes.
#
# 6-node non-colocated layout:
#   Node 0 (HEAD/Rollout):  128 SWE workers + Docker-in-Docker — NO GPUs
#   Node 1 (Inference):     4 vLLM engines × TP=2 = 8 GPUs (DP=4)
#   Nodes 2-5 (Training):   Megatron TP=4, PP=4, CP=2 = 32 GPUs (DP=1)
#
# Fully async: generation and training overlap. Workers loop continuously,
# staleness manager caps how far generation can get ahead of training.
#
# Batch: train_batch_size=128, n_samples=4 → 512 trajectories per training step
#
# Usage: bash examples/mini_swe_agent/run_mini_swe_32B_megatron_fully_async.sh [extra_hydra_overrides...]

# Enable NCCL debug logging for cross-node weight sync
# Redirect NCCL logs to separate directory to avoid flooding main training output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
mkdir -p /scratch/nccl_logs
export NCCL_DEBUG_FILE=/scratch/nccl_logs/nccl_%h_%p.log

# Reduce PyTorch CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Dynamic OPENAI_BASE_URL for multi-node setup: HEAD_NODE passed from sbatch/launch script
export OPENAI_BASE_URL="http://${HEAD_NODE:-127.0.0.1}:8001/v1"

DATA_DIR="$HOME/data/swe_gym"
CKPT_PATH="$HOME/ckpts/llm_mini_swe_32b_megatron"

# Save trajectories here for debugging
MINISWE_TRAJ_DIR="$HOME/mini_swe_agent_trajs"

# --- Inference layout (Rollout Node) ---
# 4 engines × TP=2 = 8 GPUs (DP=4)
NUM_INFERENCE_ENGINES=4
INFERENCE_TP_SIZE=2

# --- Training layout (4 Training Nodes) ---
# Megatron: TP=4, PP=4, CP=2 → DP = 32/(4*4*2) = 1
POLICY_NUM_GPUS=8
REF_NUM_GPUS=8
NNODES=4
MEGATRON_TP=4
MEGATRON_PP=4
MEGATRON_CP=2

# Gradient checkpointing (Megatron transformer_config_kwargs)
RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS=1

LOGGER=wandb

# Fully async parameters
MAX_STALENESS_STEPS=10
NUM_PARALLEL_GENERATION_WORKERS=128
MINI_BATCH_SIZE=128

# NOTE: Container must have CPATH set pointing to nvidia-cudnn and nvidia-nccl headers
# from the pre-built venv, so Ray workers' uv sync can build transformer-engine-torch.
# See sbatch_32b_megatron_async.sh for the docker run -e CPATH=... flags.
uv run --extra mcore --extra miniswe --env-file examples/mini_swe_agent/.env.miniswe \
  -m examples.mini_swe_agent.main_mini_swe_fully_async \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-32B" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$POLICY_NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$REF_NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  ++trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
  ++trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
  ++trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
  trainer.use_sample_packing=true \
  trainer.flash_attn=true \
  trainer.gradient_checkpointing=true \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=false \
  trainer.eval_interval=200 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=40960 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='0.0.0.0' \
  generator.http_endpoint_port=8002 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.85 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe" \
  trainer.run_name="mini_swe_32B_megatron_async" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  ++trainer.max_training_seq_len=40960 \
  ++generator.miniswe_config_path="examples/mini_swe_agent/swebench.yaml" \
  ++generator.miniswe_traj_dir=$MINISWE_TRAJ_DIR \
  "$@"
