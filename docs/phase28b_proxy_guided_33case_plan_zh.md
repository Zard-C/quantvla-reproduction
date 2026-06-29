# Phase 28B: Proxy-Guided 33-Case Expansion Plan

本阶段把 Phase 28A 的 15-case diagnostic 扩大到 33 个 matched task-init pair，用来判断 sensitivity-guided protection 是否只是小样本现象。

## Case List

覆盖 task 4、6、8 的 init `0..10`：

```text
4:0..10
6:0..10
8:0..10
```

这样同时包含：

- Phase 28A 的关键 repair/regression case；
- Phase 13 30-case 中暴露过的更宽 init 范围；
- task 8 init 10 这类 Phase 28A 中 speed-only 失败、proxy 修复的关键样本。

## Variants

默认跑四路：

| variant | purpose |
| --- | --- |
| FP16 baseline | 行为和速度参考 |
| speed-only compile | 最大 warm serving speedup，检验闭环误差放大 |
| proxy blocks8-15 eager | closed-loop proxy-guided protection |
| random block1 eager | 随机保护对照 |

`proxy block0` 默认不跑，避免时间过长。如果 33-case 结果需要区分 local spike proxy 和 closed-loop proxy，可以之后用 `RUN_PROXY_BLOCK0=1` 单独补跑。

## Success Criteria

我们不要求 proxy 与 baseline 行为完全一致。更合理的判据是：

1. proxy 相比 speed-only 有更好的 success / repair-regression balance；
2. proxy 相比 random protection 有更好的 net repair；
3. proxy 保留一部分 server p50 speedup；
4. remaining regressions 可以被 paired trace 解释，而不是只看总成功率。

## Commands

在 5090 workspace 运行：

```bash
cd /root/autodl-tmp/quantvla-reproduction-current
bash toy_quantvla/run_phase28b_proxy_guided_33case.sh
```

默认输出：

```text
toy_quantvla/results/phase28B_proxy_guided_33case_v1_summary.json
docs/phase28b_proxy_guided_33case_report_zh.md
```

## Expected Reading

如果结果是：

```text
speed-only fastest but lower success
proxy blocks8-15 partially recovers success
random block1 does not recover as well
```

则说明 guide 具有可扩展价值。

如果结果接近 Phase 13 的旧 30-case：

```text
proxy blocks8-15 still lower than baseline
but maybe better/worse than speed-only and random
```

也仍然有价值：它会把论文结论从“guide 可以恢复 baseline”修正为“guide 是一个风险降低器，不是行为等价保证”。
