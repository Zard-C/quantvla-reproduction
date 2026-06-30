# Phase29 Finer Duration Proxy 计划

Phase28D 说明 duration-window fallback 这条路有信息量，但单个粗窗口不够精确。`0-250` 能把 speed-only 从 `16/33` 修复到 `18/33`，但 p50 从约 `70 ms` 升到约 `89 ms`。它很可能覆盖了接近物体、对齐、接触、夹爪闭合和初始搬起这些 grasp-critical steps；后续很多 step 只是相对规则的搬运/放置，闭环敏感性可能更低。

因此 Phase29 不再问“哪个大窗口最好”，而是问：

1. `0-250` 的收益到底来自哪一段？
2. 能否用更窄窗口保留 repair，同时少牺牲速度？
3. 如果单一 duration proxy 不够，是否需要进入 layer x duration 的二阶策略？

## 已知对照

| run | success | server p50 |
| --- | --- | --- |
| FP16 baseline | `19/33` | `156.2 ms` |
| speed-only compile | `16/33` | `70.2 ms` |
| Phase28C blocks0-3 eager | `18/33` | `67.6 ms` |
| Phase28D window 0-250 | `18/33` | `88.8 ms` |

当前工程最强 tradeoff 仍是 Phase28C `blocks0-3 eager`。Phase29 的目标不是立刻替代它，而是把 duration 这一维的机制拆明白。

## 默认窗口

脚本: `toy_quantvla/run_phase29_finer_duration_proxy.sh`

默认窗口:

```text
0:120,0:180,0:220,80:240,120:280,160:240,240:320
```

这些窗口按如下假设设计：

| window | 假设 |
| --- | --- |
| `0:120` | initial approach / early alignment |
| `0:180` | approach + pre-contact |
| `0:220` | approach + contact / early grasp |
| `80:240` | 去掉最早 approach，聚焦接触和抓取 |
| `120:280` | grasp / lift centered |
| `160:240` | narrow contact-grasp bucket |
| `240:320` | post-grasp lift / early transport |

如果这轮仍然不够，可扩大到 atomic sweep：

```text
0:80,80:160,160:240,240:320,320:500,0:120,0:180,0:220,80:240,120:280
```

## 评分方式

我们不应该只最大化“像 FP16”。因为 speed-only 有时会产生 beneficial branch，例如 baseline 失败但 speed-only 成功的 case。更合适的 proxy score 是：

$$
Score(w)=|H_{repair}(w)|-|B_{regress}(w)|-\alpha |S_{regress}(w)|-\lambda \Delta latency(w)
$$

其中：

- `H`: baseline 成功但 speed-only 失败的 harmful speed regression set。
- `B`: baseline 失败但 speed-only 成功的 beneficial branch set。
- `S`: baseline 和 speed-only 都成功的 stable success set。
- `Delta latency`: 相对 speed-only 的 p50 或 eager fraction 代价。

Phase28D 给出的关键集合：

```text
H = 4:6, 6:0, 6:6, 8:3, 8:7, 8:10
B = 6:7, 6:9, 8:9
```

好窗口应该优先修复 `H`，同时避免打坏 `B`。

## 启动命令

```bash
cd /root/autodl-tmp/quantvla-reproduction-current
bash toy_quantvla/run_phase29_finer_duration_proxy.sh 2>&1 | tee phase29_tmux.log
```

建议 tmux：

```bash
tmux new-session -d -s phase29 'cd /root/autodl-tmp/quantvla-reproduction-current && set -o pipefail; bash toy_quantvla/run_phase29_finer_duration_proxy.sh 2>&1 | tee phase29_tmux.log'
```

输出：

```text
toy_quantvla/results/phase29_finer_duration_proxy_33case_v1_summary.json
docs/phase29_finer_duration_proxy_report_zh.md
```

## 判读标准

1. 如果一个窄窗口达到 `18/33` 且 p50 明显低于 `88.8 ms`，它就是比 `0-250` 更好的 duration proxy。
2. 如果所有窄窗口都低于 `0-250`，说明 early prefix 的收益来自多个阶段叠加，单段 fallback 不够。
3. 如果不同 task/init 需要不同窗口，说明 duration sensitivity 是 case-conditioned，下一步应做 layer x duration 或 task-conditioned guard。
