#!/usr/bin/env python3
"""Generate a human-readable report for the overnight recovery run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "log"
TRAIN_DIR = ROOT_DIR / "train_dir" / "sample_factory"

DEFAULT_EXP_NAME = "sf_maca_recovery_20260412_163435"
DEFAULT_REPORT_PATH = LOG_DIR / "sf_maca_recovery_night_final_report.txt"
DEFAULT_MASTER_LOG = LOG_DIR / "sf_maca_recovery_night_master.log"


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(summary: Optional[Dict], key: str) -> str:
    if not summary:
        return "n/a"
    value = summary.get(key)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def summarize_eval(title: str, payload: Optional[Dict]) -> str:
    if not payload:
        return f"{title}: missing"
    summary = payload["summary"]
    return (
        f"{title}: "
        f"win_rate={metric(summary, 'win_rate')}, "
        f"round_reward_mean={metric(summary, 'round_reward_mean')}, "
        f"opponent_round_reward_mean={metric(summary, 'opponent_round_reward_mean')}, "
        f"true_reward_mean={metric(summary, 'true_reward_mean')}, "
        f"fire_action_frac_mean={metric(summary, 'fire_action_frac_mean')}, "
        f"executed_fire_action_frac_mean={metric(summary, 'executed_fire_action_frac_mean')}, "
        f"attack_opportunity_frac_mean={metric(summary, 'attack_opportunity_frac_mean')}, "
        f"missed_attack_frac_mean={metric(summary, 'missed_attack_frac_mean')}, "
        f"course_change_frac_mean={metric(summary, 'course_change_frac_mean')}, "
        f"course_unique_frac_mean={metric(summary, 'course_unique_frac_mean')}, "
        f"episode_len_mean={metric(summary, 'episode_len_mean')}"
    )


def delta_line(title: str, before: Optional[Dict], after: Optional[Dict], key: str) -> str:
    if not before or not after:
        return f"{title}: n/a"
    before_value = before["summary"].get(key)
    after_value = after["summary"].get(key)
    if before_value is None or after_value is None:
        return f"{title}: n/a"
    delta = after_value - before_value
    return f"{title}: {before_value:.6f} -> {after_value:.6f} (delta {delta:+.6f})"


def parse_master_log(path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if not path.exists():
        return info

    patterns = {
        "resume_checkpoint": re.compile(r"resume_checkpoint=(.+)$"),
        "resume_env_steps": re.compile(r"resume_env_steps=(\d+)"),
        "run_start": re.compile(r"run_start exp=([^ ]+)"),
        "run_finished": re.compile(r"run_finished exp=([^ ]+)"),
        "latest_checkpoint": re.compile(r"latest_checkpoint=(.+)$"),
        "stage2_exploration_override": re.compile(r"stage2_exploration_override new_value=([^ ]+)"),
    }

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    info[key] = match.group(1)
    return info


def parse_stage_deltas(path: Path) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return results

    pattern = re.compile(
        r"stage_finished name=([^ ]+) start_env_steps=(\d+) end_env_steps=(\d+) env_step_delta=(\d+) latest_checkpoint=(.+)$"
    )
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.search(line)
            if not match:
                continue
            stage_name, start_env_steps, end_env_steps, env_step_delta, latest_checkpoint = match.groups()
            results[stage_name] = {
                "start_env_steps": start_env_steps,
                "end_env_steps": end_env_steps,
                "env_step_delta": env_step_delta,
                "latest_checkpoint": latest_checkpoint,
            }
    return results


def latest_checkpoint_path(exp_name: str) -> Optional[Path]:
    ckpt_dir = TRAIN_DIR / exp_name / "checkpoint_p0"
    candidates = []
    for ckpt in ckpt_dir.glob("checkpoint_*.pth"):
        try:
            stat = ckpt.stat()
        except FileNotFoundError:
            continue
        if stat.st_size <= 0:
            continue
        candidates.append((stat.st_mtime, ckpt))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def assessment(final_fixrule: Optional[Dict]) -> str:
    if not final_fixrule:
        return "Missing final fix_rule eval, cannot assess success."

    summary = final_fixrule["summary"]
    win_rate = float(summary.get("win_rate", 0.0))
    fire_frac = float(summary.get("executed_fire_action_frac_mean", summary.get("fire_action_frac_mean", 0.0)))
    episode_len = float(summary.get("episode_len_mean", 0.0))

    if win_rate >= 0.20 and fire_frac >= 0.05 and episode_len < 900:
        return "Ideal success: model is firing consistently and beating fix_rule at a meaningful rate."
    if win_rate >= 0.05 and fire_frac >= 0.02:
        return "Minimum success: model escaped the pure-evasion optimum and regained some combat ability."
    if fire_frac < 0.005:
        return "Failure: policy is still effectively in the never-fire regime."
    if win_rate == 0.0 and episode_len >= 900:
        return "Failure: policy still looks like a passive drag-out strategy."
    return "Partial result: aggression recovered somewhat, but fix_rule performance is still weak."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXP_NAME)
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--master-log", default=str(DEFAULT_MASTER_LOG))
    args = parser.parse_args()

    exp_name = args.experiment
    output_path = Path(args.output)
    master_log = Path(args.master_log)

    pre_fixrule = read_json(LOG_DIR / "sf_maca_recovery_night_pre_fixrule.eval20.json")
    pre_noatt = read_json(LOG_DIR / "sf_maca_recovery_night_pre_noatt.eval20.json")

    stage1_primary = read_json(LOG_DIR / "sf_maca_recovery_night_stage1_noatt.eval20.json")
    stage1_secondary = read_json(LOG_DIR / "sf_maca_recovery_night_stage1_noatt.vs_fix_rule.eval20.json")
    stage2_primary = read_json(LOG_DIR / "sf_maca_recovery_night_stage2_fixrule.eval20.json")
    stage2_secondary = read_json(LOG_DIR / "sf_maca_recovery_night_stage2_fixrule.vs_noatt.eval20.json")
    stage3_primary = read_json(LOG_DIR / "sf_maca_recovery_night_stage3_fixrule.eval20.json")
    stage3_secondary = read_json(LOG_DIR / "sf_maca_recovery_night_stage3_fixrule.vs_noatt.eval20.json")

    master_info = parse_master_log(master_log)
    stage_deltas = parse_stage_deltas(master_log)
    latest_ckpt = latest_checkpoint_path(exp_name)

    lines = []
    lines.append("Night Training Recovery Report")
    lines.append("============================")
    lines.append(f"Experiment: {exp_name}")
    lines.append(f"Master log: {master_log}")
    lines.append(f"Resume checkpoint: {master_info.get('resume_checkpoint', 'n/a')}")
    lines.append(f"Resume env steps: {master_info.get('resume_env_steps', 'n/a')}")
    lines.append(f"Latest checkpoint: {latest_ckpt if latest_ckpt else 'n/a'}")
    lines.append(f"Stage2 exploration override: {master_info.get('stage2_exploration_override', 'none')}")
    lines.append("")
    lines.append("Before Training")
    lines.append("---------------")
    lines.append(summarize_eval("Pre-night vs fix_rule", pre_fixrule))
    lines.append(summarize_eval("Pre-night vs fix_rule_no_att", pre_noatt))
    lines.append("")
    lines.append("Stage Results")
    lines.append("-------------")
    lines.append(summarize_eval("Stage1 primary (vs fix_rule_no_att)", stage1_primary))
    if "stage1_noatt" in stage_deltas:
        lines.append(
            "Stage1 env delta: "
            f"{stage_deltas['stage1_noatt']['start_env_steps']} -> {stage_deltas['stage1_noatt']['end_env_steps']} "
            f"(delta {stage_deltas['stage1_noatt']['env_step_delta']})"
        )
    lines.append(summarize_eval("Stage1 secondary (vs fix_rule)", stage1_secondary))
    lines.append(summarize_eval("Stage2 primary (vs fix_rule)", stage2_primary))
    if "stage2_fixrule" in stage_deltas:
        lines.append(
            "Stage2 env delta: "
            f"{stage_deltas['stage2_fixrule']['start_env_steps']} -> {stage_deltas['stage2_fixrule']['end_env_steps']} "
            f"(delta {stage_deltas['stage2_fixrule']['env_step_delta']})"
        )
    lines.append(summarize_eval("Stage2 secondary (vs fix_rule_no_att)", stage2_secondary))
    lines.append(summarize_eval("Stage3 primary (vs fix_rule)", stage3_primary))
    if "stage3_fixrule" in stage_deltas:
        lines.append(
            "Stage3 env delta: "
            f"{stage_deltas['stage3_fixrule']['start_env_steps']} -> {stage_deltas['stage3_fixrule']['end_env_steps']} "
            f"(delta {stage_deltas['stage3_fixrule']['env_step_delta']})"
        )
    lines.append(summarize_eval("Stage3 secondary (vs fix_rule_no_att)", stage3_secondary))
    lines.append("")
    lines.append("Key Deltas")
    lines.append("----------")
    lines.append(delta_line("Fix_rule win_rate", pre_fixrule, stage3_primary, "win_rate"))
    lines.append(delta_line("Fix_rule fire_action_frac_mean", pre_fixrule, stage3_primary, "fire_action_frac_mean"))
    lines.append(delta_line("Fix_rule missed_attack_frac_mean", pre_fixrule, stage3_primary, "missed_attack_frac_mean"))
    lines.append(delta_line("Fix_rule true_reward_mean", pre_fixrule, stage3_primary, "true_reward_mean"))
    lines.append(delta_line("Fix_rule episode_len_mean", pre_fixrule, stage3_primary, "episode_len_mean"))
    lines.append("")
    lines.append("Assessment")
    lines.append("----------")
    lines.append(assessment(stage3_primary))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
