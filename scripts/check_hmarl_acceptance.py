#!/usr/bin/env python
"""Acceptance checker for HMARL evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--contact_threshold", type=float, default=0.015)
    parser.add_argument("--opportunity_threshold", type=float, default=0.005)
    parser.add_argument("--win_rate_threshold", type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.eval_json)
    if not path.exists():
        raise FileNotFoundError("eval json not found: %s" % path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})

    contact = float(summary.get("contact_frac_mean", 0.0))
    opportunity = float(summary.get("attack_opportunity_frac_mean", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))

    level1 = contact >= float(args.contact_threshold)
    level2 = opportunity >= float(args.opportunity_threshold)
    level3 = win_rate >= float(args.win_rate_threshold)

    report = {
        "eval_json": str(path),
        "contact_frac_mean": contact,
        "attack_opportunity_frac_mean": opportunity,
        "win_rate": win_rate,
        "thresholds": {
            "contact": float(args.contact_threshold),
            "opportunity": float(args.opportunity_threshold),
            "win_rate": float(args.win_rate_threshold),
        },
        "levels": {
            "level1_contact_stable": bool(level1),
            "level2_attack_opportunity": bool(level2),
            "level3_win_rate": bool(level3),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    # Return non-zero if final target is not met.
    raise SystemExit(0 if level3 else 2)


if __name__ == "__main__":
    main()
