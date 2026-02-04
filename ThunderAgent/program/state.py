"""Program state management."""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..profile.state import ProfileState


class ProgramStatus(Enum):
    """Status of a program.
    
    Lifecycle:
        PAUSED -> REASONING (request starts, GPU inference)
        REASONING -> ACTING (response received, executing tool)
        ACTING -> REASONING (next request starts)
        ACTING -> PAUSED (explicitly paused)
        * -> STOPPED (program released)
    """
    REASONING = "reasoning"  # On GPU, running inference
    ACTING = "acting"        # Off GPU, executing tool
    PAUSED = "paused"        # Waiting (not actively processing)
    STOPPED = "stopped"      # Program completed/released


@dataclass
class ProgramState:
    """State of a single program (task)."""
    backend_url: str  # Which backend this program is assigned to
    status: ProgramStatus = ProgramStatus.PAUSED
    context_len: int = 0
    total_tokens: int = 0
    step_count: int = 0
    # True iff the tool execution for the previous step has finished and the program
    # is ready to issue the next vLLM request (becoming REASONING).
    tool_finished: bool = True
    # Unix timestamp when the program transitioned to ACTING (used for time-decay weighting).
    acting_since: Optional[float] = None
    profile: Optional["ProfileState"] = None  # Profile timing data (when profiling enabled)
    waiting_event: Optional[asyncio.Event] = field(default=None, repr=False)  # Event to wait on when paused
    marked_for_pause: bool = False  # Mark REASONING program to pause when it becomes ACTING
