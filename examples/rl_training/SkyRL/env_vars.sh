# env_vars.sh
# Centralized environment configuration for SkyRL Docker containers

# Base Scratch Directory (mapped to /scratch inside container)
export SCRATCH_DIR="/scratch"
export HOME="$SCRATCH_DIR"

# HuggingFace - Cache models and datasets
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$SCRATCH_DIR/hf_cache"

# Triton - Compilation cache (Critical for permission errors)
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton_cache"

# FlashInfer - JIT compilation cache
export FLASHINFER_WORKSPACE_DIR="$SCRATCH_DIR/flashinfer_cache"

# UV / Python - Package cache
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/uv_python"

# XDG Base Directories - Redirects ~/.cache, ~/.config, ~/.local/share
export XDG_CACHE_HOME="$SCRATCH_DIR/cache"
export XDG_CONFIG_HOME="$SCRATCH_DIR/config"
export XDG_DATA_HOME="$SCRATCH_DIR/data"

# Application Specific
export DATA="$SCRATCH_DIR"
export MINISWE_TRAJ_DIR="$SCRATCH_DIR/mini_swe_agent_trajs_32B"
export MSWEA_COST_TRACKING='ignore_errors'

# NCCL Settings - Increase timeout for colocated training+inference weight sync
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=1800
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_ASYNC_ERROR_HANDLING=1

# Ensure directories exist and are writable
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$MINISWE_TRAJ_DIR"

# Set open permissions to avoid issues if container user changes
chmod 777 "$TRITON_CACHE_DIR" 2>/dev/null || true
chmod 777 "$XDG_CONFIG_HOME" 2>/dev/null || true
