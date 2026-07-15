# Phase44-45 计划：hybrid BO follow-up queue

## 背景

Phase43 正在跑 10-case hybrid probe：

```text
duration window x layer eager island
```

为了避免 GPU 空跑，Phase43 结束后自动进入两段 follow-up。

## Phase44: hybrid held-out validation

Phase44 读取 Phase43 summary，自动选择 follow-up tactics：

```text
fp16
speed_only
window_0_20
window_2_12
+ up to 3 best blocks* hybrid candidates
```

选择规则：

1. paired regression 少的优先；
2. 同 regression 下优先 speed-feasible；
3. 再比较 success rate、speedup、repairs。

默认 cases：

```text
tasks 0/1/4/6/8 x init 35/36/37 = 15 cases
```

脚本：

```bash
bash toy_quantvla/run_phase44_n17_hybrid_heldout_auto.sh
```

## Phase45: all-task stress check

Phase45 读取 Phase44 summary，再自动选择最多 6 个 tactic，扩展到全部 LIBERO-10 task 的小规模 stress check。

默认 cases：

```text
tasks 0-9 x init 38/39 = 20 cases
```

脚本：

```bash
bash toy_quantvla/run_phase45_n17_hybrid_alltask_stress_auto.sh
```

## 队列脚本

如果 Phase43 已经在 tmux `phase43_hybrid_bo` 中运行，可以另开一个等待队列：

```bash
tmux new-session -d -s phase43_followup_queue \
  'bash toy_quantvla/run_phase43_followup_queue.sh 2>&1 | tee phase43_followup_queue_tmux.log'
```

队列逻辑：

```text
wait for phase43_hybrid_bo
-> run Phase44 auto held-out
-> run Phase45 auto all-task stress
```

## 预期价值

Phase43 是 candidate-generation probe，Phase44 是新 init held-out，Phase45 是跨 task stress。这样能把 BO 扩展从“发现候选”推进到“候选是否跨 init、跨 task 仍然有效”。
