#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

SEVERITY_ORDER = {"LOW": 1, "MODERATE": 2, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def max_severity(vulns_json):
    max_level = 0
    # pip-audit JSON schema: list of {"name","version","vulns":[{"id","fix_versions", "aliases", "severity"? , "advisory"?}]}
    for pkg in vulns_json:
        for v in pkg.get("vulns", []) or []:
            sev = (v.get("severity") or "").upper()
            if not sev:
                # Try advisory severity
                adv = v.get("advisory") or {}
                sev = (adv.get("severity") or "").upper()
            # Fallback: CRITICAL if CVSS score >= 9, HIGH if >= 7
            score = None
            try:
                score = float((v.get("cvss") or {}).get("score", 0))
            except Exception:
                score = None
            if not sev and score is not None:
                sev = "CRITICAL" if score >= 9 else ("HIGH" if score >= 7 else "MEDIUM")
            level = SEVERITY_ORDER.get(sev, 2)
            if level > max_level:
                max_level = level
    return max_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--requirements", default="requirements.txt")
    parser.add_argument("--fail-on-high", action="store_true")
    args = parser.parse_args()

    code, out, err = run([sys.executable, "-m", "pip_audit", "-r", args.requirements, "-f", "json"])
    # pip-audit exits non-zero if any vulns; we need JSON regardless
    try:
        data = json.loads(out or "[]")
    except json.JSONDecodeError:
        print(err or out)
        sys.exit(1)

    level = max_severity(data)
    if args.fail_on_high and level >= SEVERITY_ORDER["HIGH"]:
        print("pip-audit found HIGH/CRITICAL vulnerabilities")
        print(json.dumps(data, indent=2))
        sys.exit(1)
    # Otherwise pass
    print("pip-audit completed; no HIGH/CRITICAL vulnerabilities")


if __name__ == "__main__":
    main()
