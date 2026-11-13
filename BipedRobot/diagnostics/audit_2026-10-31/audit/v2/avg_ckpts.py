"""Average a policy's weights over several saved checkpoints. Usage: avg_ckpts.py OUT_PT IN_PT [IN_PT ...]"""
import sys

import torch

states = [torch.load(p, map_location="cpu") for p in sys.argv[2:]]
out = dict(states[-1])
out["actor"] = {k: (torch.stack([s["actor"][k] for s in states]).mean(0) if states[0]["actor"][k].is_floating_point() else states[0]["actor"][k]) for k in states[0]["actor"]}
out["averaged_from"] = sys.argv[2:]
torch.save(out, sys.argv[1])
