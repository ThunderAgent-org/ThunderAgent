"""Sandbox tool — a ProgramAwareTool backed by Docker containers."""
import logging
from typing import List, Optional

from .program_aware_tool import ProgramAwareTool

logger = logging.getLogger(__name__)


class Sandbox(ProgramAwareTool):
    """Docker-container-based sandbox whose lifecycle is tied to a program.

    Inherits all tracking from ProgramAwareTool; tool_ids are Docker container ids.

    Cleanup logic:
        - Stop and remove each container.
        - If the container's image has no other containers left, remove the image too.
    """

    def __init__(self, program_id: str, container_ids: Optional[List[str]] = None) -> None:
        super().__init__(program_id, tool_ids=container_ids)

    def cleanup(self) -> List[str]:
        """Stop and remove all containers. Remove images with no remaining containers.

        Returns:
            List of container_ids that were cleaned up.
        """
        try:
            import docker
            client = docker.from_env()
        except Exception as e:
            logger.warning(f"[{self.program_id}] Cannot connect to Docker daemon: {e}")
            container_ids = list(self.tool_ids)
            self.tool_ids.clear()
            return container_ids

        container_ids = list(self.tool_ids)
        # Track image_id for each container so we can check if image should be removed
        image_ids = set()

        # Step 1: Stop and remove containers
        for cid in container_ids:
            try:
                container = client.containers.get(cid)
                image_id = container.image.id
                image_ids.add(image_id)
                container.stop()
                container.remove()
                logger.info(f"[{self.program_id}] Stopped and removed container {cid[:12]}")
            except Exception as e:
                logger.warning(f"[{self.program_id}] Failed to cleanup container {cid[:12]}: {e}")

        # Step 2: Remove images that have no remaining containers
        for image_id in image_ids:
            try:
                remaining = client.containers.list(all=True, filters={"ancestor": image_id})
                if not remaining:
                    client.images.remove(image_id, force=True)
                    logger.info(f"[{self.program_id}] Removed image {image_id[:12]} (no containers left)")
                else:
                    logger.debug(
                        f"[{self.program_id}] Kept image {image_id[:12]} "
                        f"({len(remaining)} container(s) still using it)"
                    )
            except Exception as e:
                logger.warning(f"[{self.program_id}] Failed to remove image {image_id[:12]}: {e}")

        self.tool_ids.clear()
        return container_ids

    def __repr__(self) -> str:
        return f"Sandbox(program_id={self.program_id!r}, container_ids={list(self.tool_ids)})"
