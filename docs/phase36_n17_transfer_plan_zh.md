# Phase 36: GR00T N1.7 Transfer Validation Plan

## 目标

Phase36 不再继续量化 kernel 路线，而是验证本文提出的
Closed-Loop Sensitivity-Guided Tactic Search 是否能迁移到新的权重。

核心问题：

> 换成 GR00T N1.7 LIBERO checkpoint 后，speed-only、duration fallback、
> eager island 这些 tactic 的 speed-risk Pareto 关系是否仍然需要重新估计？

这不是证明 `0--120` 是 universal tactic。相反，Phase36 应该验证：

1. 新权重上的 FP16 baseline 是否和 N1.5 有不同 fragile slice。
2. speed-only compile 是否仍然产生 paired repairs/regressions。
3. `0--120` 是否仍然是好 proxy，还是需要重新搜索 duration window。
4. CLSG-TS 能否用小规模 probe 找到更稳的 tactic frontier。

## 模型和场景

- model: GR00T N1.7 LIBERO checkpoint
- default local model path: `/root/autodl-tmp/models/GR00T-N1.7-LIBERO/libero_10`
- default Isaac-GR00T source: `/root/autodl-tmp/Isaac-GR00T`
- task suite: `libero_10`
- checkpoint denoising steps: `4` (`num_inference_timesteps` in N1.7 config)
- rollout execution horizon: `8` env actions per policy call (`n_action_steps`)
- official env prefix: `libero_sim/...`
- embodiment tag: `LIBERO_PANDA`

Phase36 首轮不换仿真平台。先保持 LIBERO eval loop 不变，只换模型/权重和上游源码版本。

## 阶段

### Phase36a: environment and model smoke

检查项：

1. N1.7 Isaac-GR00T 源码能在当前 `gr00t-libero-py310` 环境 import。
2. N1.7 LIBERO checkpoint 能用 `Gr00tPolicy` 加载。
3. 官方 `gr00t/eval/run_gr00t_server.py` 能启动 server。
4. 官方 `gr00t/eval/rollout_policy.py` 能完成 1-case smoke。

默认命令：

```bash
bash toy_quantvla/run_phase36_n17_official_smoke.sh
```

默认环境：

```text
libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
```

默认只跑 `1` 个 episode，目标是验证：

- N1.7 checkpoint 能加载；
- N1.7 official policy server 能启动；
- LIBERO gymnasium wrapper 能创建环境；
- official rollout client 能通过 ZMQ 调用 policy 并完成 episode。

注意：N1.7 main 分支的 policy/eval API 与 N1.5 不同。Phase36a 先跑 official FP16 smoke；通过后再写 N1.7 专用 timed/compiled wrapper，把 CLSG-TS 候选接入新 API。

当前新增两个入口：

- `toy_quantvla/run_phase36_n17_official_smoke.sh`
  - 使用 N1.7 官方 `run_gr00t_server.py` 和 `rollout_policy.py`。
  - 只验证 checkpoint、server、LIBERO env、ZMQ client 能闭环跑通。
  - 默认 `RECORD_VIDEO=0`，避免 ffmpeg/video encoding 影响 smoke 和后续 latency。
- `toy_quantvla/run_phase36_n17_timed_rollout.sh`
  - 使用 `toy_quantvla/n17_timed_gr00t_server.py` 作为 server。
  - 保留官方 rollout client，但在 server 侧记录 latency / memory / request trace。
  - 支持 `TORCH_COMPILE_TARGET=action_head_model` 和 request-index duration fallback。
  - 用于 Phase36b/36c 的 tactic probe，不用于取代 official smoke。

### Phase36a 当前状态

5090 实例上已经完成 N1.7 official smoke：

```text
tag: phase36_n17_official_smoke_1case_v8
model: /root/autodl-tmp/models/GR00T-N1.7-LIBERO/libero_10
env: libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
n_episodes: 1
seed: 20260705
record_video: false
client_status: 0
success_rate: 1.0
episode_length: 269
episode_reward: 1.0
```

结果文件：

- `toy_quantvla/results/phase36_n17_official_smoke_1case_v8_summary.json`
- `toy_quantvla/results/phase36_n17_official_smoke_1case_v8_client.log`
- `toy_quantvla/results/phase36_n17_official_smoke_1case_v8_server.log`

这说明当前远端环境已经能完成：

1. N1.7 checkpoint 加载；
2. official GR00T policy server 启动；
3. LIBERO gymnasium env 创建；
4. official rollout client 通过 ZMQ 完整调用 policy；
5. 一个无视频录制干扰的闭环 episode 成功结算。

### 远端 Isaac-GR00T 临时补丁

由于 `nvidia/Cosmos-Reason2-2B` 是 gated model，当前账号能通过 HuggingFace
认证，但没有该 gated repo 的文件下载权限。因此 5090 上的 Isaac-GR00T 源码做了
四个最小实验补丁：

1. `gr00t/model/modules/qwen3_backbone.py`
   - 增加 `GR00T_QWEN3_INIT_FROM_CONFIG=1` 路径。
   - 使用公开的 `Qwen/Qwen3-VL-2B-Instruct` config 构建 `Qwen3VLForConditionalGeneration`。
   - 再由本地 N1.7 checkpoint shards 加载实际权重。
2. `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`
   - 增加 `GR00T_QWEN3_PROCESSOR_NAME` / `GR00T_QWEN3_CONFIG_NAME`。
   - processor 从公开 Qwen snapshot 读取，避免访问 gated Cosmos processor。
3. `gr00t/eval/rollout_policy.py`
   - 增加 `GR00T_DISABLE_VIDEO_RECORDING=1`。
   - 修复官方逻辑里 `video_dir=None` 仍自动生成 `/tmp/sim_eval_videos_*` 的行为。
   - 增加 `GR00T_POLICY_CLIENT_TIMEOUT_MS`。
   - 避免 `torch.compile` 首请求 cold compile 超过官方 `PolicyClient` 默认 15s timeout。
4. `gr00t/eval/sim/LIBERO/libero_env.py`
   - 增加 `GR00T_LIBERO_USE_BENCHMARK_INIT_STATES=1` 路径。
   - 默认行为不变：`seed` 只作为 robosuite seed。
   - 打开该开关后，`seed` 被解释成 LIBERO benchmark init index，并调用
     `task_suite.get_task_init_states(task_id)` + `env.set_init_state(...)`。
   - 这使 Phase36b 的 `task × init` 口径与 N1.5 实验一致。

当前使用的公开 snapshot：

```text
/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
```

运行 smoke / timed rollout 时需要设置：

```bash
export GR00T_QWEN3_INIT_FROM_CONFIG=1
export GR00T_QWEN3_CONFIG_NAME=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
export GR00T_QWEN3_PROCESSOR_NAME=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
export GR00T_POLICY_CLIENT_TIMEOUT_MS=600000
```

注意：这些是远端上游源码补丁，不是本文方法本身的一部分。它们只是为了绕开
gated base model metadata 的下载限制，关闭默认视频录制带来的非推理开销，并让
验证集严格复用 LIBERO 官方初始状态。其中 benchmark init-state 补丁只在
`GR00T_LIBERO_USE_BENCHMARK_INIT_STATES=1` 时启用。

### Phase36b bridge: timed 1-case smoke

在进入 15-case tactic discovery 之前，先完成了 timed server 的 1-case 桥接验证。

| Run | Tactic | Status | Success | Requests | Prepare | All p50 | All mean | All max | Hot p50 | Hot mean | Memory reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase36_n17_timed_fp16_1case_v1` | FP16 eager baseline | complete | 1/1 | 33 | 14.80s | 105.3ms | 139.4ms | 890.6ms | 102.4ms | 116.0ms | 6188 MiB |
| `phase36_n17_speedonly_1case_v2` | compile `action_head.model` | complete | 1/1 | 34 | 16.35s | 74.1ms | 682.9ms | 20493.4ms | 72.5ms | 82.6ms | 6312 MiB |
| `phase36_n17_window_0_10_1case_v1` | request window `0--10` eager, then compiled | complete | 1/1 | 38 | 19.52s | 89.9ms | 278.7ms | 6428.0ms | 79.3ms | 91.1ms | 6216 MiB |

这里 `Hot` 对 FP16 和 speed-only 指去掉第一个 policy request 后重算；对
`window_0_10` 指去掉第一个 request 和第一次 compiled request 后重算。对
speed-only 来说，第一个 request 包含 inductor cold compile，耗时约 `20.49s`。
对 `window_0_10` 来说，request 11 是首次从 eager 切到 compiled，耗时约 `6.43s`。
因此：

1. speed-only 的热态 p50 约从 `102.4ms` 降到 `72.5ms`，单 case 上约 `1.41x`；
2. request-window fallback 机制已验证：`window_0_10` 记录到 `10` 次 eager request 和
   `28` 次 compiled request；
3. cold compile 成本非常明显，必须与热态 latency 分开报告；
4. 没有 `GR00T_POLICY_CLIENT_TIMEOUT_MS=600000` 时，speed-only v1 会在首请求触发
   `zmq.error.Again`，因为官方 `PolicyClient` 默认 15s timeout 太短。

结果文件：

- `toy_quantvla/results/phase36_n17_timed_fp16_1case_v1_*`
- `toy_quantvla/results/phase36_n17_speedonly_1case_v2_*`
- `toy_quantvla/results/phase36_n17_window_0_10_1case_v1_*`

### Phase36b exact-init bridge

在进入 `task × init` probe 前，先验证 N1.7 wrapper 能严格使用 LIBERO
benchmark init states，而不是仅使用随机 seed。

```text
tag: phase36_n17_exactinit_task0_init21_fp16_1case_v2
task: libero_10 task 0
init index: 21
tactic: FP16 eager baseline
LIBERO_USE_BENCHMARK_INIT_STATES: 1
success: 1/1
episode_length: 259
episode_reward: 1.0
policy requests: 33
prepare: 15.82s
server latency p50 / mean / max: 100.97ms / 143.45ms / 818.34ms
memory reserved: 6188 MiB
```

`v1` 暴露出一个 wrapper 细节：official vector env 在 episode 结束后会额外调用
一次 `reset(seed=None)`。因此 benchmark init-state 补丁必须只在 `seed` 非空时
调用 `set_init_state()`；无 seed 的收尾 reset 仍走官方默认 reset。修正后 `v2`
正常完成，说明后续 Phase36b 可以用 `SEED=21/22/23` 表示真实 benchmark init
index。

结果文件：

- `toy_quantvla/results/phase36_n17_exactinit_task0_init21_fp16_1case_v2_*`

### Phase36b: small tactic discovery

如果 Phase36a 通过，扩展到 15 cases：

```text
tasks 0/1/4/6/8 × init 21/22/23
```

这里的 `init 21/22/23` 必须通过
`LIBERO_USE_BENCHMARK_INIT_STATES=1` 启用。否则 N1.7 官方 wrapper 的
`--seed` 只是随机种子，不会调用 `set_init_state()`，实验语义会退化成
`task × seed` 而不是 `task × benchmark init`。

候选：

- FP16 baseline
- speed-only compile
- request window 0--10
- request window 0--20
- request window 0--30
- request window 10--30
- request window 20--50

注意：N1.7 timed server 当前的 duration fallback index 是 policy request index，
不是环境 step。由于 `n_action_steps=8`，一个 `720` env-step episode 最多约 `90`
次 policy request。因此不能直接沿用 N1.5 里的 `0--120` / `0--180` 这类窗口，
否则会覆盖几乎整个 episode，失去 duration 分辨率。

目标是判断 N1.7 的敏感 request window 是否仍集中在早期，以及 speed-only 的
hot latency 收益是否值得承担 cold compile 和闭环风险。

### Phase36c: 30-case held-out validation

如果 Phase36b 找到 Pareto 候选，再跑 30 cases：

```text
tasks 0--9 × init 21/22/23
```

候选：

- FP16 baseline
- speed-only compile
- best duration window
- best behavior-first combo

输出应进入 multi-fold selector，但不要和 N1.5 folds 直接混成同一个模型的结论。
论文中应作为 cross-weight transfer evidence 单独报告。

## 判读规则

Phase36 的核心判读不是“哪个 tactic 单折最高”，而是：

1. speed-only 是否仍然最快但带来更高 paired regression。
2. duration fallback 是否能降低 regression 或提高 worst-slice success。
3. behavior-first 和 speed-constrained 目标是否选出不同 tactic。
4. N1.7 的敏感 window/layer 是否和 N1.5 相同。

如果 N1.7 仍然需要重新估计 tactic frontier，则支持论文主张：

> For a new VLA checkpoint, acceleration tactics should be selected by
> closed-loop sensitivity-guided validation rather than assumed from another
> checkpoint.

如果 N1.7 复现相似的 early-window pattern，则进一步说明：

> Early receding-horizon control steps may be repeatedly overrepresented in
> closed-loop sensitivity, but the exact tactic still requires validation.
