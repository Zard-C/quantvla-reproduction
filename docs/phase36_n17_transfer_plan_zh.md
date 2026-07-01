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
- `toy_quantvla/run_phase36_n17_timed_rollout.sh`
  - 使用 `toy_quantvla/n17_timed_gr00t_server.py` 作为 server。
  - 保留官方 rollout client，但在 server 侧记录 latency / memory / request trace。
  - 支持 `TORCH_COMPILE_TARGET=action_head_model` 和 request-index duration fallback。
  - 用于 Phase36b/36c 的 tactic probe，不用于取代 official smoke。

### Phase36b: small tactic discovery

如果 Phase36a 通过，扩展到 15 cases：

```text
tasks 0/1/4/6/8 × init 21/22/23
```

候选：

- FP16 baseline
- speed-only compile
- duration 0--80
- duration 0--120
- duration 0--180
- duration 80--240

目标是判断 N1.7 的敏感 duration 是否仍集中在早期。

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
