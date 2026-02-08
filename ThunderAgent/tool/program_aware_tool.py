"""Program-aware tool management.

A ProgramAwareTool is a tool whose lifecycle is shared with an LLM agentic program.
When the program starts, its tools are registered; when the program ends, its tools
are cleaned up together.
"""
import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class ProgramAwareTool:
    """Tools that shares the same life cycle with the program.

    Usage:
        - Agent sends tool_ids (e.g. docker container ids) via extra_body.
        - ThunderAgent creates a ProgramAwareTool for the program.
        - When the program is released, all associated tools are cleaned up.
    """

    def __init__(self, program_id: str, tool_ids: Optional[List[str]] = None) -> None:
        self.program_id = program_id
        self.tool_ids: Set[str] = set(tool_ids) if tool_ids else set()

    def register(self, tool_ids: List[str]) -> None:
        """Register tool_ids for this program. Duplicate ids are ignored."""
        for tid in tool_ids:
            if tid not in self.tool_ids:
                self.tool_ids.add(tid)
                logger.info(f"[{self.program_id}] Registered tool {tid[:12]}")

    def unregister(self, tool_ids: List[str]) -> None:
        """Unregister tool_ids from this program."""
        for tid in tool_ids:
            if tid in self.tool_ids:
                self.tool_ids.discard(tid)
                logger.info(f"[{self.program_id}] Unregistered tool {tid[:12]}")

    def cleanup(self) -> List[str]:
        """Remove and return all tool_ids (for external cleanup).

        Returns:
            List of tool_ids that were associated with this program.
        """
        tool_ids = list(self.tool_ids)
        self.tool_ids.clear()
        if tool_ids:
            logger.info(
                f"[{self.program_id}] Cleaned up {len(tool_ids)} tool(s): "
                f"{[t[:12] for t in tool_ids]}"
            )
        return tool_ids

    def __len__(self) -> int:
        return len(self.tool_ids)

    def __repr__(self) -> str:
        return f"ProgramAwareTool(program_id={self.program_id!r}, tool_ids={self.tool_ids})"
