import subprocess
import sys
from pathlib import Path

TESTS = [
    {
        "name": "Forward null v1",
        "cmd": ["python", "scripts/null_tests/forward_null_v1.py"],
        "must_contain": ["mean", "P(C814>C390)", "max ratio"],
        "status": "OFFICIAL PASS",
    },
    {
        "name": "Clumpy source stress test",
        "cmd": ["python", "scripts/null_tests/forward_null_v2_clumpy_source.py"],
        "must_contain": ["P(C814>C390)", "max ratio"],
        "status": "STRESS TEST",
    },
    {
        "name": "Non-lens control null",
        "cmd": ["python", "scripts/null_tests/forward_null_v3_nonlens_controls.py"],
        "must_contain": ["N valid", "max ratio"],
        "status": "OFFICIAL PASS",
    },
    {
        "name": "CMB blackbody consistency",
        "cmd": ["python", "scripts/cosmology/cmb_blackbody_consistency_test.py"],
        "must_contain": ["CMB BLACKBODY CONSISTENCY TEST", "Saved"],
        "status": "CONSTRAINT / REVIEW",
    },
    {
        "name": "BAO consistency",
        "cmd": ["python", "scripts/cosmology/bao_consistency_test.py"],
        "must_contain": ["BAO CONSISTENCY TEST"],
        "status": "CONSTRAINT / REVIEW",
    },
]

def run_test(test):
    print("=" * 80)
    print(f"TEST: {test['name']}")
    print(f"STATUS: {test['status']}")
    print("-" * 80)

    proc = subprocess.run(
        test["cmd"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = proc.stdout
    print(output)

    ok = proc.returncode == 0
    for token in test["must_contain"]:
        if token not in output:
            ok = False
            print(f"[MISSING EXPECTED TEXT] {token}")

    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return ok, output

def main():
    Path("reproducibility_logs").mkdir(exist_ok=True)
    log_path = Path("reproducibility_logs/reproducibility_check.log")

    all_ok = True
    full_log = []

    for test in TESTS:
        ok, output = run_test(test)
        all_ok = all_ok and ok
        full_log.append("=" * 80)
        full_log.append(f"TEST: {test['name']}")
        full_log.append(f"STATUS: {test['status']}")
        full_log.append(output)
        full_log.append(f"RESULT: {'PASS' if ok else 'FAIL'}")

    log_path.write_text("\n".join(full_log), encoding="utf-8")

    print("=" * 80)
    print(f"Saved log: {log_path}")
    print(f"OVERALL RESULT: {'PASS' if all_ok else 'REVIEW REQUIRED'}")

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
