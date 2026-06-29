"""Analyze Phase 28A paired flips for proxy-guided mixed precision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

TAG_PREFIX = "phase28A_proxy_guided_15case_v1"
SUMMARY_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
OUT_MD = DOCS / "phase28_proxy_guided_flip_analysis_zh.md"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def run_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["name"]: run for run in data["runs"]}


def episode_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {ep["case"]: ep for ep in run["episodes"]}


def outcome(ep: dict[str, Any] | None) -> str:
    if ep is None:
        return "-"
    return ("S" if ep["success"] else "F") + str(ep["steps"])


def find_pair(data: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    for pair in data["paired_comparisons"]:
        if pair["left"] == left and pair["right"] == right:
            return pair
    raise KeyError(f"missing pair {right} vs {left}")


def signed_delta(run: dict[str, Any], baseline: dict[str, Any]) -> str:
    delta = int(run["total_successes"]) - int(baseline["total_successes"])
    if delta > 0:
        return f"+{delta}"
    return str(delta)


def speedup(run: dict[str, Any]) -> str:
    value = run.get("speedup_vs_baseline_server_p50")
    if value is None:
        return "-"
    return f"{value}x"


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def classify_cases(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    baseline = episode_map(runs["baseline"])
    speed = episode_map(runs["speed_only"])
    block0 = episode_map(runs["proxy_block0"])
    blocks8 = episode_map(runs["proxy_blocks8_15"])
    random = episode_map(runs["random_block1"])

    interesting = ["4:6", "6:0", "8:10", "8:9", "4:9", "4:7"]
    notes = {
        "4:6": "speed-only 打坏 FP16 成功；三个保护策略都救回，说明这里对 full compile 边界敏感。",
        "6:0": "speed-only/regression；block0 和 blocks8-15 救回，random 没救回，更支持 proxy 有信息量。",
        "8:10": "speed-only/regression；两个 proxy 救回，random 没救回，是 proxy-guided 的强证据之一。",
        "8:9": "baseline 失败、speed-only 成功；blocks8-15 保留了这个 beneficial branch，block0/random 没保住。",
        "4:9": "baseline 和 speed-only 都成功，但三个 eager-protection 版本都失败，说明保护本身也是策略扰动。",
        "4:7": "只有 random 成功且接近 horizon，像低 margin / 偶然分支，不宜作为主证据。",
    }
    rows = []
    for case in interesting:
        rows.append(
            [
                case,
                outcome(baseline.get(case)),
                outcome(speed.get(case)),
                outcome(block0.get(case)),
                outcome(blocks8.get(case)),
                outcome(random.get(case)),
                notes[case],
            ]
        )
    return rows


def build_report(data: dict[str, Any]) -> str:
    runs = run_map(data)
    baseline = runs["baseline"]
    speed = runs["speed_only"]
    block0 = runs["proxy_block0"]
    blocks8 = runs["proxy_blocks8_15"]
    random = runs["random_block1"]

    frontier_rows = []
    for name in ["baseline", "speed_only", "proxy_block0", "proxy_blocks8_15", "random_block1"]:
        run = runs[name]
        pair = None if name == "baseline" else find_pair(data, "baseline", name)
        frontier_rows.append(
            [
                run["label"],
                f"{run['total_successes']}/{run['total_episodes']}",
                signed_delta(run, baseline),
                fmt_ms(run.get("server_p50_ms")),
                speedup(run),
                "-" if pair is None else len(pair["repaired"]),
                "-" if pair is None else len(pair["regressed"]),
                "-" if pair is None else pair["net"],
            ]
        )

    speed_pair_rows = []
    for name in ["proxy_block0", "proxy_blocks8_15", "random_block1"]:
        run = runs[name]
        pair = find_pair(data, "speed_only", name)
        speed_pair_rows.append(
            [
                run["label"],
                f"{run['total_successes']}/{run['total_episodes']}",
                fmt_ms(run.get("server_p50_ms")),
                speedup(run),
                len(pair["repaired"]),
                len(pair["regressed"]),
                pair["net"],
                ", ".join(pair["repaired"]) or "-",
                ", ".join(pair["regressed"]) or "-",
            ]
        )

    lines = [
        "# Phase 28A Flip Analysis: Proxy-Guided Mixed Precision",
        "",
        "这份报告只分析已完成的 15 个 matched task-init pair。它的目标不是证明最终部署配置已经收敛，而是回答一个更具体的问题：论文里的 sensitivity-guided 设计思想，是否能把 speed-only 编译造成的闭环退化拉回来。",
        "",
        "## 速度-成功率前沿",
        "",
        md_table(
            [
                "run",
                "success",
                "delta vs baseline",
                "server p50 ms",
                "speedup",
                "repairs vs baseline",
                "regressions vs baseline",
                "net",
            ],
            frontier_rows,
        ),
        "",
        "这里最重要的对照是 `speed_only` 和 `proxy_blocks8_15`。`speed_only` 把 server p50 从 `84.76 ms` 降到 `50.35 ms`，但成功数从 `7/15` 掉到 `5/15`。`proxy_blocks8_15` 牺牲一部分速度，把 p50 放到 `67.36 ms`，成功数恢复到 `7/15`。",
        "",
        "## 相对 speed-only 的修复",
        "",
        md_table(
            [
                "run",
                "success",
                "server p50 ms",
                "speedup",
                "repaired",
                "regressed",
                "net",
                "repaired cases",
                "regressed cases",
            ],
            speed_pair_rows,
        ),
        "",
        "`proxy_blocks8_15` 是这轮最像“按 guide 起效”的配置：它相对 `speed_only` 修复了 `4:6, 6:0, 8:10` 三个失败，只新增 `4:9` 一个失败，净 `+2`。`random_block1` 速度相近，但净收益为 `0`；这说明并不是随便保留一个 eager block 就能稳定改善闭环表现。",
        "",
        "`proxy_block0` 也有价值，但证据更复杂。它几乎保留了 speed-only 的速度，并修复 `4:6, 6:0, 8:10`，但丢掉了 speed-only 在 `8:9` 上的 beneficial branch，同时也打坏 `4:9`。这支持一个更细的结论：same-observation spike proxy 能找到局部数值敏感边界，但不一定等价于闭环 outcome proxy。",
        "",
        "## 关键病例",
        "",
        md_table(
            ["case", "baseline", "speed_only", "block0", "blocks8_15", "random", "解读"],
            classify_cases(runs),
        ),
        "",
        "## 对论文 claim 的含义",
        "",
        "1. 这不是单纯的数值加速问题。`speed_only` 和 `proxy_blocks8_15` 都是无损语义的编译边界选择，但闭环成功率不同，说明实现边界也会成为 policy perturbation。",
        "2. 这支持 `not all layers are equal`：保护 `blocks8-15` 的收益明显高于随机 `block1`，并且能保住 `8:9` 这个 speed-only repair。",
        "3. 这也支持 `not all proxies are equal`：局部数值 spike proxy 和闭环 repair/regression proxy 给出的选择不同，后者更接近我们真正关心的 rollout outcome。",
        "4. 当前样本只有 15 个 matched cases，结论应该写成 evidence / diagnostic result，而不是最终显著性结论。",
        "",
        "## 下一步",
        "",
        "1. 把 `4:6, 6:0, 8:10, 8:9, 4:9` 做 first-divergence / trajectory branching，对比 speed-only 和 proxy_blocks8_15。",
        "2. 把样本扩到 30 或 50 个 matched cases，优先覆盖 task 4、6、8 的 init 段。",
        "3. 分离 cold compile、warm serving latency 和显存峰值，避免把部署准备时间混进在线推理速度。",
        "",
        "## 输入产物",
        "",
        f"- Summary JSON: `{SUMMARY_JSON.relative_to(ROOT)}`",
        f"- Phase 28A report: `docs/phase28_proxy_guided_mixed_precision_report_zh.md`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = load_summary()
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
