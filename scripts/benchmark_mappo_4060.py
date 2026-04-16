#!/usr/bin/env python
"""Short benchmark sweep for tuning MaCA MAPPO on a 4060-class laptop."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))


TRAIN_LOG_RE = re.compile(r"fps=([0-9]+(?:\.[0-9]+)?)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=str, default="log/mappo_4060_benchmark.json")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opponent", type=str, default="fix_rule")
    parser.add_argument("--train_dir", type=str, default="train_dir/bench")
    parser.add_argument("--steps_per_run", type=int, default=4096)
    parser.add_argument("--log_every_sec", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    return parser.parse_args()


def load_variants() -> List[Dict[str, int]]:
    return [
        {"name": "balanced_baseline", "num_envs": 8, "num_workers": 4, "rollout": 128, "hidden_size": 256},
        {"name": "cpu_push", "num_envs": 10, "num_workers": 5, "rollout": 128, "hidden_size": 256},
        {"name": "turbo_12x6", "num_envs": 12, "num_workers": 6, "rollout": 128, "hidden_size": 256},
        {"name": "turbo_14x7", "num_envs": 14, "num_workers": 7, "rollout": 128, "hidden_size": 256},
        {"name": "turbo_12x6_longroll", "num_envs": 12, "num_workers": 6, "rollout": 192, "hidden_size": 256},
    ]


def gpu_monitor(samples: List[Dict[str, float]], stop_event: threading.Event) -> None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(cmd, cwd=str(ROOT_DIR), text=True, stderr=subprocess.DEVNULL).strip()
            if output:
                util_gpu, util_mem, mem_used, mem_total = [float(v.strip()) for v in output.split(",")[:4]]
                samples.append(
                    {
                        "gpu_util": util_gpu,
                        "mem_util": util_mem,
                        "mem_used_mib": mem_used,
                        "mem_total_mib": mem_total,
                    }
                )
        except Exception:
            pass
        stop_event.wait(1.0)


def summarize_gpu(samples: List[Dict[str, float]]) -> Dict[str, float]:
    if not samples:
        return {}
    keys = samples[0].keys()
    summary = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        summary["%s_mean" % key] = float(sum(values) / len(values))
        summary["%s_max" % key] = float(max(values))
    return summary


def run_variant(args, variant: Dict[str, int]) -> Dict[str, object]:
    exp_name = "bench_%s_%d" % (variant["name"], int(time.time()))
    train_steps = max(args.steps_per_run, variant["num_envs"] * variant["rollout"] * 2)
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "maca-py37-min",
        "python",
        "scripts/train_mappo_maca.py",
        "--experiment=%s" % exp_name,
        "--train_dir=%s" % args.train_dir,
        "--device=%s" % args.device,
        "--seed=%d" % args.seed,
        "--num_envs=%d" % variant["num_envs"],
        "--num_workers=%d" % variant["num_workers"],
        "--rollout=%d" % variant["rollout"],
        "--chunk_len=16",
        "--burn_in=8",
        "--train_for_env_steps=%d" % train_steps,
        "--ppo_epochs=%d" % args.ppo_epochs,
        "--num_mini_batches=2",
        "--learning_rate=3e-4",
        "--hidden_size=%d" % variant["hidden_size"],
        "--role_embed_dim=8",
        "--course_embed_dim=16",
        "--save_every_sec=999999",
        "--log_every_sec=%d" % args.log_every_sec,
        "--tensorboard=false",
        "--eval_every_env_steps=0",
        "--eval_episodes=0",
        "--maca_opponent=%s" % args.opponent,
        "--maca_max_step=1000",
        "--maca_max_visible_enemies=4",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "%s:%s%s%s" % (
        str(ROOT_DIR),
        str(ENV_DIR),
        ":" if env.get("PYTHONPATH") else "",
        env.get("PYTHONPATH", ""),
    )

    stop_event = threading.Event()
    gpu_samples: List[Dict[str, float]] = []
    monitor_thread = threading.Thread(target=gpu_monitor, args=(gpu_samples, stop_event), daemon=True)
    monitor_thread.start()

    start_time = time.time()
    process = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stop_event.set()
    monitor_thread.join(timeout=2.0)

    output = process.stdout
    fps_values = [float(match.group(1)) for match in TRAIN_LOG_RE.finditer(output)]
    return {
        "variant": variant,
        "experiment": exp_name,
        "return_code": int(process.returncode),
        "wall_time_sec": float(time.time() - start_time),
        "fps_mean": float(sum(fps_values) / len(fps_values)) if fps_values else 0.0,
        "fps_max": float(max(fps_values)) if fps_values else 0.0,
        "fps_samples": fps_values,
        "gpu_summary": summarize_gpu(gpu_samples),
        "tail": output[-4000:],
    }


def rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        results,
        key=lambda item: (
            float(item.get("fps_mean", 0.0)),
            float(item.get("gpu_summary", {}).get("mem_used_mib_max", 0.0)),
        ),
        reverse=True,
    )


def main():
    args = parse_args()
    results = []
    for variant in load_variants():
        print("[bench] running %s" % variant["name"], flush=True)
        result = run_variant(args, variant)
        results.append(result)
        print(
            "[bench] done %s rc=%d fps_mean=%.1f gpu_util_mean=%.1f mem_used_max=%.0f"
            % (
                variant["name"],
                int(result["return_code"]),
                float(result["fps_mean"]),
                float(result["gpu_summary"].get("gpu_util_mean", 0.0)),
                float(result["gpu_summary"].get("mem_used_mib_max", 0.0)),
            ),
            flush=True,
        )

    ranked = rank_results(results)
    payload = {
        "ranked": ranked,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload["ranked"][:3], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
