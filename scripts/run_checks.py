#!/usr/bin/env python3
import subprocess
import sys

commands = [
    [sys.executable, "-m", "pytest", "-q"],
]

failed = False
for cmd in commands:
    print("Running:", " ".join(cmd))
    code = subprocess.call(cmd)
    if code != 0:
        failed = True

sys.exit(1 if failed else 0)
