"""CPU-only regression tests for pending pause accounting."""
import unittest

from ThunderAgent.backend.vllm_metrics import VLLMCacheConfig
from ThunderAgent.program.state import Program, ProgramState, ProgramStatus
from ThunderAgent.scheduler.router import MultiBackendRouter


class PendingPauseCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = MultiBackendRouter(["http://backend.invalid"])
        self.backend = self.router.get_default_backend()
        self.backend.metrics_client.cache_config = VLLMCacheConfig(
            block_size=1, num_gpu_blocks=10_000
        )

    async def asyncTearDown(self):
        await self.router.client.aclose()

    def program(self, name, tokens, status=ProgramStatus.REASONING):
        program = Program(
            program_id=name, backend_url=self.backend.url,
            total_tokens=tokens, status=status, step_count=2,
        )
        self.router.programs[name] = program
        self.backend.register_program(name, program)
        return program

    async def test_marks_only_enough_running_programs(self):
        small = self.program("small", 2_000)
        large = self.program("large", 8_500)
        await self.router._pause_until_safe(self.backend)
        self.assertTrue(small.marked_for_pause)
        self.assertFalse(large.marked_for_pause)
        self.assertEqual(self.backend.future_paused_tokens, 2_000)
        self.assertEqual(self.backend.capacity_overflow(include_future_release=True), 0)
        # Marking is a deferred release, not permission to admit another request.
        self.assertEqual(self.backend.remaining_capacity(), -700)
        self.assertFalse(self.backend.has_capacity(extra_tokens=100, extra_count=1))
        self.assertIsNone(self.router._select_backend_for_new_program(100))
        await self.router._pause_until_safe(self.backend)
        self.assertFalse(large.marked_for_pause)

    async def test_marks_additional_program_when_one_release_is_insufficient(self):
        small = self.program("small", 2_000)
        medium = self.program("medium", 3_000)
        large = self.program("large", 8_500)
        await self.router._pause_until_safe(self.backend)
        self.assertTrue(small.marked_for_pause)
        self.assertTrue(medium.marked_for_pause)
        self.assertFalse(large.marked_for_pause)
        self.assertEqual(self.backend.future_paused_tokens, 5_000)

    async def test_acting_programs_still_pause_before_running_programs(self):
        acting = self.program("acting", 2_000, ProgramStatus.ACTING)
        running = self.program("running", 8_500)
        await self.router._pause_until_safe(self.backend)
        self.assertEqual(acting.state, ProgramState.PAUSED)
        self.assertFalse(running.marked_for_pause)
        self.assertEqual(self.backend.future_paused_tokens, 0)

    async def test_pending_release_tracks_streaming_growth(self):
        marked = self.program("marked", 2_000)
        self.program("other", 8_500)
        self.router._mark_program_for_pause(marked.program_id, marked)
        self.router.update_program_tokens_streaming(marked, 1_000)
        self.assertEqual(self.backend.future_paused_tokens, 3_000)
        await self.router._pause_until_safe(self.backend)
        self.assertFalse(self.router.programs["other"].marked_for_pause)

    async def test_completion_does_not_consume_another_pending_release(self):
        first = self.program("first", 2_000)
        second = self.program("second", 3_000)
        self.router._mark_program_for_pause(first.program_id, first)
        self.router._mark_program_for_pause(second.program_id, second)
        self.router.update_program_after_request(first.program_id, first, 2_500)
        self.assertEqual(first.state, ProgramState.PAUSED)
        self.assertEqual(self.backend.future_paused_tokens, 3_000)
        self.assertFalse(first.marked_for_pause)

    async def test_release_does_not_consume_another_pending_release(self):
        first = self.program("first", 2_000)
        second = self.program("second", 3_000)
        self.router._mark_program_for_pause(first.program_id, first)
        self.router._mark_program_for_pause(second.program_id, second)
        self.router.update_program_tokens_streaming(first, 500)
        await self.router.release_program(first.program_id)
        self.assertEqual(self.backend.future_paused_tokens, 3_000)

    async def test_new_growth_can_require_another_pause(self):
        small = self.program("small", 2_000)
        large = self.program("large", 8_500)
        self.router._mark_program_for_pause(small.program_id, small)
        self.router.update_program_tokens_streaming(large, 2_000)
        await self.router._pause_until_safe(self.backend)
        self.assertTrue(large.marked_for_pause)

    async def test_repeated_mark_is_idempotent(self):
        program = self.program("marked", 2_000)
        self.router._mark_program_for_pause(program.program_id, program)
        self.router._mark_program_for_pause(program.program_id, program)
        self.assertEqual(self.backend.future_paused_tokens, 2_000)

    async def test_within_capacity_does_not_pause_anything(self):
        program = self.program("running", 9_000)
        await self.router._pause_until_safe(self.backend)
        self.assertFalse(program.marked_for_pause)


if __name__ == "__main__":
    unittest.main()
