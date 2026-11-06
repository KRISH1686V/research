"""Copy a run's resolved config with the floor friction set to the owner's lab floor, for testing only. Nothing else changes.

Usage: lab_rc.py IN_YAML OUT_YAML [LOW HIGH]
The owner reports a lab floor friction of about 0.3 to 0.4, with PLA feet. The simulator's contact friction equals the floor's value (checked on foot contacts), so the number applies directly.
"""
import sys
from pathlib import Path

import yaml

src, out = sys.argv[1:3]
low, high = (float(x) for x in sys.argv[3:5]) if len(sys.argv) > 3 else (0.3, 0.4)
config = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
config["resolved_env_config"]["domain_randomization"]["friction_range"] = [low, high]
Path(out).write_text(yaml.safe_dump(config), encoding="utf-8")
