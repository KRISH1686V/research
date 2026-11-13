#!/usr/bin/env python3
"""Read-only servo temperature/voltage watcher. NO torque, nothing is written to the servos.
Run next to robot_test.py, with servo_bridge stopped:
    python3 temp_watch.py --biped-ws ~/biped_ws --seconds 30
Gently wiggle the cables and tap the servos while it runs. It prints any reading that looks wrong."""
import argparse, sys, time
from robot_test import find_hardware_code

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--hz", type=float, default=50)
    p.add_argument("--warn-c", type=float, default=50, help="print readings at or above this")
    p.add_argument("--biped-ws", default=None)
    p.add_argument("--port", default=None); p.add_argument("--baud", type=int, default=None)
    p.add_argument("--calibration", default=None); p.add_argument("--hardware-config", default=None)
    p.add_argument("--servos-config", default=None); p.add_argument("--mock", action="store_true")
    a = p.parse_args()
    find_hardware_code(a.biped_ws)
    from biped_hardware.tools.common import setup
    cfg, bus, arr = setup(a)
    stats, bad, missed, loops = {}, 0, 0, 0
    t_end = time.time() + a.seconds
    print("Reading only, no torque. Wiggle the bus cables gently now.")
    try:
        while time.time() < t_end:
            t0 = time.time()
            st = arr.read_status(); loops += 1
            for name, s in st.items():
                if s is None:
                    missed += 1; continue
                lo, hi, v = stats.get(name, (999, -999, 99))
                stats[name] = (min(lo, s.temp_c), max(hi, s.temp_c), min(v, s.voltage_v))
                if s.temp_c >= a.warn_c:
                    bad += 1
                    print(f"  t={a.seconds-(t_end-time.time()):5.1f}s  {name}: temp {s.temp_c} C, {s.voltage_v:.1f} V   <-- suspicious")
            time.sleep(max(0, 1 / a.hz - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        try: arr.set_torque(False)
        except Exception: pass
    print(f"\n{loops} reads. Missed replies: {missed}. Suspicious readings: {bad}")
    print(f"{'joint':22}{'min C':>7}{'max C':>7}{'min V':>7}")
    for n, (lo, hi, v) in stats.items():
        print(f"{n:22}{lo:>7}{hi:>7}{v:>7.1f}")

if __name__ == "__main__":
    main()
