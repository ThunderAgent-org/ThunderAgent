from __future__ import annotations

from sweagent.utils.profiling import ProfilingAggregator


def test_profiling_aggregator_summary():
    aggregator = ProfilingAggregator()
    dump = {
        "problem_id": "test",
        "status": "success",
        "timings": {"agent_setup": 1.0, "env_preparation": 2.0},
        "steps": [
            {
                "prefill_tokens": 10,
                "decode_tokens": 5,
                "llm_reasoning_time": 0.5,
                "llm_prefill_time": 0.2,
                "llm_decode_time": 0.3,
                "tool_execution_time": 1.5,
                "observation_time": 0.2,
                "tool_types": ["bash"],
            }
        ],
        "request_totals": {
            "agent_setup_time": 1.0,
            "env_prepare_time": 2.0,
            "total_llm_reasoning_time": 0.5,
            "total_llm_prefill_time": 0.2,
            "total_llm_decode_time": 0.3,
            "total_tool_execution_time": 1.5,
            "total_observation_time": 0.2,
            "total_prefill_tokens": 10,
            "total_decode_tokens": 5,
        },
        "metadata": {
            "tokens_sent": 10,
            "tokens_received": 5,
            "api_calls": 1,
            "instance_cost": 0.1,
        },
    }
    aggregator.add(dump)
    summary = aggregator.summary()
    assert summary is not None
    aggregate = summary["aggregate"]
    assert aggregate["timings_total"]["agent_setup"] == 1.0
    assert aggregate["request_totals_total"]["total_prefill_tokens"] == 10
    assert aggregate["request_totals_total"]["total_llm_prefill_time"] == 0.2
    assert aggregate["request_totals_total"]["total_llm_decode_time"] == 0.3
    assert aggregate["step_metrics_total"]["prefill_tokens"] == 10
    assert aggregate["step_metrics_total"]["llm_prefill_time"] == 0.2
    assert aggregate["step_metrics_total"]["llm_decode_time"] == 0.3
    assert aggregate["tool_usage_counts"]["bash"] == 1