# manager_core.py
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Any
import time

class ProgramStatus(Enum):
    CREATED = "created"
    RUNNING = "running"     # Docker 正在执行命令 (Acting)
    THINKING = "thinking"   # 等待 LLM 响应 (Docker 空闲)
    PAUSED = "paused"       # Docker 被暂停
    CHECKPOINTED = "checkpointed" # Docker 被冻结到磁盘
    FINISHED = "finished"

@dataclass
class ProgramContext:
    """代表一个需要解决的问题实例 (Program)"""
    instance_id: str        # 唯一标识 (SWE-bench instance id)
    container_id: str       # 关联的 Docker 容器 ID
    image_tag: str          # 使用的 Docker 镜像
    status: ProgramStatus
    last_active: float      # 最后活跃时间戳
    
    # 资源快照 (用于监控)
    cpu_usage: float = 0.0
    mem_usage: float = 0.0

class DockerManagerBase(ABC):
    """底层 Docker 操作的抽象接口"""
    
    @abstractmethod
    def pause_container(self, container_id: str) -> bool:
        pass

    @abstractmethod
    def unpause_container(self, container_id: str) -> bool:
        pass

    @abstractmethod
    def checkpoint_container(self, container_id: str) -> bool:
        pass

    @abstractmethod
    def restore_container(self, container_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_container_stats(self, container_id: str) -> Dict[str, Any]:
        """获取当前 CPU/Mem 信息"""
        pass

class SchedulerBase(ABC):
    """资源调度策略抽象接口"""
    
    @abstractmethod
    def on_llm_request_start(self, program: ProgramContext):
        """当 Agent 开始请求 LLM (进入 Thinking)"""
        pass

    @abstractmethod
    def on_llm_request_end(self, program: ProgramContext):
        """当 LLM 返回结果 (准备进入 Acting)"""
        pass
    
    @abstractmethod
    def register_program(self, instance_id: str, container_id: str, image_tag: str):
        """注册一个新的 Program 实例"""
        pass