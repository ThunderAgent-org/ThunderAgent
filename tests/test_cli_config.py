"""Regression tests for the 233b63f import-order bug.

The router must be built from the CLI config, not from Config() defaults.
Each check runs in a fresh interpreter because the bug is about import order.
"""
import json
import subprocess
import sys

DRIVER = r'''
import json, sys, uvicorn

def fake_run(app, **kwargs):  # stand-in for uvicorn.run: inspect, don't serve
    r = sys.modules["ThunderAgent.app"].router
    print(json.dumps({
        "scheduling": r.scheduling_enabled,
        "backends": list(r.backends),
        "backend_client": sorted({type(b.metrics_client).__name__ for b in r.backends.values()}),
        "profile": r.profile_enabled,
        "interval": r._scheduler_interval,
        "weight": r.acting_token_weight,
        "decay": sorted({b.use_acting_token_decay for b in r.backends.values()}),
    }))

uvicorn.run = fake_run
sys.argv = ["thunderagent", "--router", "default", "--backends", "http://a:1,http://b:2",
            "--backend-type", "sglang", "--profile", "--scheduler-interval", "2.5",
            "--acting-token-weight", "0.5", "--use-acting-token-decay"]
from ThunderAgent.__main__ import main
main()
'''


def _last_line(code: str) -> str:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return proc.stdout.strip().splitlines()[-1]


def test_importing_package_does_not_import_app():
    code = "import sys, ThunderAgent; print('ThunderAgent.app' in sys.modules)"
    assert _last_line(code) == "False"


def test_cli_options_reach_router():
    assert json.loads(_last_line(DRIVER)) == {
        "scheduling": False,
        "backends": ["http://a:1", "http://b:2"],
        "backend_client": ["SGLangMetricsClient"],
        "profile": True,
        "interval": 2.5,
        "weight": 0.5,
        "decay": [True],
    }


def test_public_api_still_importable():
    code = "from ThunderAgent import register_routes, get_program_id; print('ok')"
    assert _last_line(code) == "ok"
