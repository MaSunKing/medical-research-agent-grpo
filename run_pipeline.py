"""Public offline entry point; does not launch a deployment or contact providers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "check"))
    args = parser.parse_args()
    if args.command == "demo":
        sys.path.insert(0, str(ROOT / "examples"))
        from offline_demo import run
        print(json.dumps(run(), indent=2, allow_nan=False))
        return
    suites = {
        "agent": ["test_recent_runtime_contracts"],
        "training": ["test_joint", "test_authority", "test_v16_regressions",
                     "test_v17_regressions", "test_v18_regressions"],
        "shared": [p.stem for p in sorted((ROOT / "shared").glob("test_*.py"))],
    }
    for directory, tests in suites.items():
        subprocess.run([sys.executable, "-B", "-m", "unittest", *tests],
                       cwd=ROOT / directory, check=True)
    subprocess.run([sys.executable, "-B", "verify_package.py"],
                   cwd=ROOT / "judge", check=True)
    subprocess.run([sys.executable, "-B", "run_pipeline.py", "demo"],
                   cwd=ROOT, check=True)
    print(json.dumps({"status": "passed", "scope": "offline_core_contracts",
                      "api_calls": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
