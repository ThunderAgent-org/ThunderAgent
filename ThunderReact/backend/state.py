"""Backend state management."""
import asyncio
import logging
from typing import List, Optional, Set, TYPE_CHECKING

import httpx

from .vllm_metrics import VLLMMetrics, VLLMCacheConfig

if TYPE_CHECKING:
    from ..program.state import ProgramState

logger = logging.getLogger(__name__)

# Keep only the most recent N metrics samples
METRICS_HISTORY_SIZE = 12

# Buffer tokens reserved per program for decode phase
DECODE_BUFFER = 1024


class BackendState:
    """State of a single VLLM backend with self-managed metrics monitoring."""
    
    def __init__(self, url: str):
        self.url = url
        self.healthy = True
        self.metrics_history: List[VLLMMetrics] = []
        
        # Static cache config (fetched once at startup)
        self.cache_config: Optional[VLLMCacheConfig] = None
        
        # Program token tracking
        self.active_program_tokens: int = 0   # REASONING + ACTING programs only
        self.active_program_count: int = 0    # Number of REASONING + ACTING programs
        self.total_program_tokens: int = 0    # All non-STOPPED programs (including PAUSED)
        
        # Paused programs waiting to be resumed
        self.paused_programs: Set[str] = set()  # program_id set
        
        # Programs marked to be paused when they transition REASONING -> ACTING
        self.marked_for_pause: Set[str] = set()  # program_id set
        
        # Capacity blocked flag: when True, all new requests must wait
        self.capacity_blocked: bool = False
        self.capacity_unblocked_event: asyncio.Event = asyncio.Event()
        
        # Capacity check task
        self._capacity_check_task: Optional[asyncio.Task] = None
        
        # Metrics monitoring (self-managed)
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_stop = False
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def completions_url(self) -> str:
        """Chat completions API endpoint."""
        return f"{self.url}/v1/chat/completions"
    
    @property
    def metrics_url(self) -> str:
        """Prometheus metrics endpoint."""
        return f"{self.url}/metrics"
    
    @property
    def latest_metrics(self) -> Optional[VLLMMetrics]:
        """Get the most recent metrics sample."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    @property
    def active_program_tokens_ratio(self) -> float:
        """Ratio of active program tokens to total capacity."""
        if not self.cache_config or self.cache_config.total_tokens_capacity == 0:
            return 0.0
        return self.active_program_tokens / self.cache_config.total_tokens_capacity
    
    # -------------------------------------------------------------------------
    # Capacity Check
    # -------------------------------------------------------------------------
    
    def has_capacity(self, extra_tokens: int = 0, extra_count: int = 0) -> bool:
        """Check if adding extra tokens/programs would exceed capacity.
        
        Constraint: active_tokens + active_count * DECODE_BUFFER <= total_capacity
        """
        if not self.cache_config:
            return True  # No config, assume ok
        
        tokens = self.active_program_tokens + extra_tokens
        count = self.active_program_count + extra_count
        required = tokens + count * DECODE_BUFFER
        return required <= self.cache_config.total_tokens_capacity
    
    def capacity_overflow(self) -> int:
        """Return how many tokens we're over capacity (0 if within capacity)."""
        if not self.cache_config:
            return 0
        required = self.active_program_tokens + self.active_program_count * DECODE_BUFFER
        overflow = required - self.cache_config.total_tokens_capacity
        return max(0, overflow)
    
    # -------------------------------------------------------------------------
    # Program Token Management
    # -------------------------------------------------------------------------
    
    def add_active_program(self, tokens: int) -> None:
        """Add a program to active (REASONING/ACTING) state."""
        self.active_program_tokens += tokens
        self.active_program_count += 1
    
    def remove_active_program(self, tokens: int) -> None:
        """Remove a program from active state (-> PAUSED)."""
        self.active_program_tokens -= tokens
        self.active_program_count -= 1
        if self.active_program_tokens < 0:
            logger.warning(f"active_program_tokens went negative ({self.active_program_tokens}), resetting to 0")
            self.active_program_tokens = 0
        if self.active_program_count < 0:
            logger.warning(f"active_program_count went negative ({self.active_program_count}), resetting to 0")
            self.active_program_count = 0
    
    def add_total_program(self, tokens: int) -> None:
        """Add tokens to total (new program, any state except STOPPED)."""
        self.total_program_tokens += tokens
    
    def remove_total_program(self, tokens: int) -> None:
        """Remove tokens from total (program STOPPED)."""
        self.total_program_tokens -= tokens
        if self.total_program_tokens < 0:
            logger.warning(f"total_program_tokens went negative ({self.total_program_tokens}), resetting to 0")
            self.total_program_tokens = 0
    
    def update_program_tokens(self, old_tokens: int, new_tokens: int) -> None:
        """Update tokens when a program's total_tokens changes (e.g., after request)."""
        diff = new_tokens - old_tokens
        self.active_program_tokens += diff
        self.total_program_tokens += diff
        if self.active_program_tokens < 0:
            logger.warning(f"active_program_tokens went negative ({self.active_program_tokens}), resetting to 0")
            self.active_program_tokens = 0
        if self.total_program_tokens < 0:
            logger.warning(f"total_program_tokens went negative ({self.total_program_tokens}), resetting to 0")
            self.total_program_tokens = 0
    
    # Legacy methods for compatibility
    def add_program_tokens(self, tokens: int) -> None:
        """Add tokens when a program becomes active (REASONING/ACTING)."""
        self.add_active_program(tokens)
    
    def remove_program_tokens(self, tokens: int) -> None:
        """Remove tokens when a program becomes inactive (PAUSED/STOPPED)."""
        self.remove_active_program(tokens)
    
    # -------------------------------------------------------------------------
    # Metrics Monitoring (self-managed)
    # -------------------------------------------------------------------------
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start background metrics monitoring for this backend."""
        if self._monitor_task is not None:
            return  # Already running
        
        self._monitor_stop = False
        self._client = httpx.AsyncClient(timeout=10.0)
        
        # Fetch cache config once at startup
        await self.fetch_cache_config()
        
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started metrics monitoring for {self.url} (interval: {interval}s, kv_capacity: {self.cache_config.total_tokens_capacity if self.cache_config else 'unknown'} tokens)")
    
    async def stop_monitoring(self):
        """Stop background metrics monitoring."""
        if self._monitor_task is None:
            return
        
        self._monitor_stop = True
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        self._monitor_task = None
        
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info(f"Stopped metrics monitoring for {self.url}")
    
    async def _monitor_loop(self, interval: float):
        """Background loop to periodically fetch metrics."""
        while not self._monitor_stop:
            try:
                await self._fetch_metrics()
            except Exception as e:
                logger.debug(f"Error fetching metrics from {self.url}: {e}")
            await asyncio.sleep(interval)
    
    async def fetch_cache_config(self) -> bool:
        """Fetch static cache config from vLLM.
        
        Can be called independently of start_monitoring().
        Uses existing client if monitoring, otherwise creates a temporary one.
        """
        client = self._client
        close_client = False
        
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True
        
        try:
            resp = await client.get(self.metrics_url)
            if resp.status_code == 200:
                self.cache_config = VLLMCacheConfig.from_prometheus_text(resp.text)
                logger.info(f"Fetched cache config for {self.url}: block_size={self.cache_config.block_size}, num_gpu_blocks={self.cache_config.num_gpu_blocks}, total_capacity={self.cache_config.total_tokens_capacity}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to fetch cache config from {self.url}: {e}")
            return False
        finally:
            if close_client:
                await client.aclose()
    
    async def _fetch_metrics(self) -> bool:
        """Fetch and update metrics from vLLM /metrics endpoint."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code == 200:
                metrics = VLLMMetrics.from_prometheus_text(resp.text)
                self.metrics_history.append(metrics)
                # Keep only the most recent samples
                if len(self.metrics_history) > METRICS_HISTORY_SIZE:
                    self.metrics_history = self.metrics_history[-METRICS_HISTORY_SIZE:]
                self.healthy = True
                return True
            else:
                self.healthy = False
                return False
        except Exception as e:
            logger.debug(f"Failed to fetch metrics from {self.url}: {e}")
            self.healthy = False
            return False
    
    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        result = {
            "url": self.url,
            "healthy": self.healthy,
            "monitoring": self._monitor_task is not None,
            "active_program_tokens": self.active_program_tokens,
            "active_program_count": self.active_program_count,
            "active_program_tokens_ratio": round(self.active_program_tokens_ratio, 4),
            "total_program_tokens": self.total_program_tokens,
            "paused_program_count": len(self.paused_programs),
            "marked_for_pause_count": len(self.marked_for_pause),
            "capacity_blocked": self.capacity_blocked,
            "decode_buffer": DECODE_BUFFER,
            "capacity_overflow": self.capacity_overflow(),
        }
        # Include cache config (static)
        if self.cache_config:
            result["cache_config"] = {
                "block_size": self.cache_config.block_size,
                "num_gpu_blocks": self.cache_config.num_gpu_blocks,
                "total_tokens_capacity": self.cache_config.total_tokens_capacity,
            }
        # Include latest metrics (dynamic)
        if self.metrics_history:
            latest = self.latest_metrics
            result["metrics"] = {
                "num_requests_running": latest.num_requests_running,
                "num_requests_waiting": latest.num_requests_waiting,
                "kv_cache_usage_perc": round(latest.kv_cache_usage_perc, 4),
                "prefix_cache_hit_rate": round(latest.prefix_cache_hit_rate, 4),
                "prefix_cache_queries": latest.prefix_cache_queries,
                "prefix_cache_hits": latest.prefix_cache_hits,
                "prompt_tokens_total": latest.prompt_tokens_total,
                "generation_tokens_total": latest.generation_tokens_total,
                "num_preemptions": latest.num_preemptions,
                "requests_completed": latest.total_requests_completed,
                "last_updated": latest.timestamp,
                "history_size": len(self.metrics_history),
            }
        return result
