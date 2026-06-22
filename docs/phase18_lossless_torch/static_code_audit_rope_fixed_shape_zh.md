# Phase 18 静态代码审计：RoPE 与固定 shape 无损加速

## 结论

本地复现仓库不包含 Isaac-GR00T 的完整模型实现，所以这里的结论来自本地 wrapper 代码、已有 5090 profiler 结果，以及 NVIDIA/Isaac-GR00T 官方 `n1.5-release` 分支静态审计。这个阶段不重新跑 GPU，只做“瞪眼法”筛选，目标是找出下一轮 5090 profiler 应该盯住的位置。

当前判断：

- N1.5 action head 里没有 RoPE。它用的是 diffusion timestep sinusoidal embedding、DiT block 的 sinusoidal positional embedding、以及 action sequence 的 learned position embedding。
- RoPE 在 Eagle text backbone 内部。N1.5 配置里 text model 是 `Qwen3ForCausalLM`，`rope_theta=1000000`，`use_cache=false`，实际 RoPE 实现由安装环境里的 Hugging Face Transformers Qwen3 代码决定。
- 最值得先查的不是 client 侧 `tolist()`，而是 server 侧三类东西：RoPE/cos/sin 是否每 request 重算、action head denoising loop 里固定张量是否每 step 重建、以及 final `normalized_action.cpu()` 是否引入可见同步。

## 版本边界

我们当前实验线是 N1.5/Eagle，不是 N1.7/Qwen3-VL/Cosmos-Reason2。

N1.7 main 分支已经在 `qwen3_backbone.py` 里显式处理 rotary `inv_freq` 重置问题，说明 RoPE 正确性和 cache 在上游确实是一个被认真处理的问题。但这不能直接套到 N1.5：N1.5 的 Eagle wrapper 调用的是 `Qwen3ForCausalLM`，不是 N1.7 的新 backbone 封装。

## RoPE 位置

N1.5 的 `eagle2_hg_model/config.json` 显示：

- `text_config.architectures = ["Qwen3ForCausalLM"]`
- `text_config.model_type = "qwen3"`
- `text_config.rope_theta = 1000000`
- `text_config.rope_scaling = null`
- `text_config.use_cache = false`
- `_attn_implementation = "flash_attention_2"`

这意味着 RoPE 主要发生在 Qwen3 language model attention 内部。Isaac-GR00T 的 Eagle wrapper 自己只负责把 vision token 写入 language input embedding，再调用 language model。

下一轮 5090 上必须用实际安装环境确认：

```bash
python - <<'PY'
import inspect
import transformers
import transformers.models.qwen3.modeling_qwen3 as q

print("transformers", transformers.__version__)
print("file", q.__file__)
for name in ["Qwen3RotaryEmbedding", "Qwen3Attention", "apply_rotary_pos_emb"]:
    obj = getattr(q, name, None)
    print("\n==", name, "==")
    if obj is not None:
        print(inspect.getsource(obj)[:4000])
PY
```

重点看三件事：

- `cos/sin` 是不是每次 forward 都根据 `position_ids` 重新算。
- `inv_freq` 或频率表有没有 buffer/cache，是否已经在模型加载后固定。
- `use_cache=false` 是否让 prefill-like 推理每次都完整重算，而没有任何 KV/position 复用空间。

## Action Head 固定张量

`FlowmatchingActionHead.get_action` 的 denoising loop 每次 request 通常跑 8 step。静态看，每一步都重复构造或查表：

- `timesteps_tensor = torch.full(...)`
- `action_encoder(... timesteps_tensor ...)`
- `SinusoidalPositionalEncoding.forward` 里按 hidden dim 生成 `torch.arange`、`torch.tensor(10000.0)`、`exp`、`sin`、`cos`
- action position 的 `torch.arange(action_features.shape[1])`
- `position_embedding(pos_ids)`
- `future_tokens.weight.unsqueeze(0).expand(...)`
- `torch.cat((state_features, future_tokens, action_features), dim=1)`

这些操作大概率不是最大瓶颈，但它们满足三个条件：固定 shape、固定 denoising schedule、无损可缓存。因此很适合作为低风险 patch：

1. request 内预先生成 8 个 `timesteps_tensor`。
2. request 内预先生成 action `pos_ids` 和 `pos_embs`。
3. request 内预先生成 `future_tokens`。
4. request 内预先拼好 static prefix：`state_features + future_tokens`，每 step 只拼 `action_features`。
5. 给 `SinusoidalPositionalEncoding` 加 device/dtype/hidden_dim 级别的频率 buffer，避免每 step 创建 `arange` 和 `tensor(10000.0)`。

精确性门槛：同一个 normalized observation、同一个 RNG seed、同一个 dtype 下，`action_pred` 必须 `max_abs_diff = 0` 或至少 bitwise equal。因为这里改的是 deterministic constant path，理论上可以做到完全一致。

## DiT 内部重复计算

`cross_attention_dit.py` 里还有两类重复：

- `TimestepEncoder.forward` 每个 DiT forward 都调用 Diffusers `Timesteps` 和 `TimestepEmbedding`。denoising schedule 固定时，`temb` 也可以按 step 缓存。
- `DiT.forward` 每次都执行 `hidden_states.contiguous()` 和 `encoder_hidden_states.contiguous()`。如果输入本来 contiguous，这只是轻量 no-op；如果触发 copy，`encoder_hidden_states` 在 denoising loop 内不变，应该搬到 loop 外或缓存一次。

这两项不一定显著，但 profiler 上应该能直接看到：

- `aten::contiguous`
- `aten::copy_`
- `aten::_to_copy`
- `aten::arange`
- `aten::sin`
- `aten::cos`
- `aten::cat`

## Server D2H 细节

之前说 `env.step(action.tolist())` 不是 server GPU D2H，这个判断仍然成立：LIBERO simulator client 拿到的已经是 NumPy/action dict。

但 N1.5 官方 `policy.py` 里确实有 server 侧 final copy：

- model 输出 `model_pred["action_pred"].float()`
- `_get_unnormalized_action` 调用 `normalized_action.cpu()`
- 然后走 transform 的 `unapply`

这个 copy 是 API 返回 NumPy/CPU action 前的必要步骤，最终无法完全取消。但它可能造成同步，所以 profiler 要量化它的真实占比。考虑到 action tensor 很小，如果这里超过 1 ms，通常不是带宽问题，而是前面 kernel queue 被这个 `.cpu()` 强制同步暴露出来。

## Eagle Backbone 审计点

N1.5 Eagle wrapper 中，vision path 的配置是 `select_layer=-1`，所以 `extract_feature` 对 SigLIP vision model 调用 `output_hidden_states=False`。这个路径没有明显“为了取中间层而存全量 hidden states”的浪费。

Language path 的外层 `EagleBackbone.forward_eagle` 明确调用 `self.eagle_model(... output_hidden_states=True, return_dict=True)`，然后取 `hidden_states[self.select_layer]`。初始化时它会 pop 掉不用的后续 LLM layers，所以如果实际 `select_layer` 正好对应截断后最后一层，那么有机会改成 `output_hidden_states=False` 并取 `last_hidden_state`，避免保存所有 retained hidden states。

这个点可能比 action-head 小修小补更有价值，但风险也更高：

- 必须先打印实际 `backbone_cfg.select_layer` 和截断后 layer 数量。
- 必须确认 `last_hidden_state` 和原来的 `hidden_states[self.select_layer]` 数值完全一致。
- 必须确认不同 task description 长度下 shape 不变或 prewarm 覆盖。

5090 上建议先做固定 observation A/B：

```bash
python - <<'PY'
# 伪代码：在加载后的 policy.model.backbone 上确认截断层数和 select_layer。
backbone = policy.model.backbone
print("select_layer", backbone.select_layer)
print("num_layers_after_pop", len(backbone.eagle_model.language_model.model.layers))
PY
```

## 5090 Profiler Checklist

先不改代码，直接用固定 observation 做一次低扰动 profiler。目标不是测 rollout success，而是找 server `get_action` 内哪些固定路径重复出现。

```bash
cd /root/autodl-tmp/quantvla-reproduction

# 先定位实际安装源码，避免看错版本。
python - <<'PY'
import inspect
import gr00t.model.policy as p
import gr00t.model.action_head.flow_matching_action_head as f
import gr00t.model.action_head.cross_attention_dit as d
import transformers.models.qwen3.modeling_qwen3 as q

for m in [p, f, d, q]:
    print(m.__name__, inspect.getfile(m))
PY

# 静态 grep。
python - <<'PY'
import inspect
import gr00t, pathlib
root = pathlib.Path(inspect.getfile(gr00t)).parents[1]
print(root)
PY
rg -n "rope|RoPE|rotary|Rotary|apply_rotary|position_ids|cache_position|cos_cached|sin_cached|inv_freq|freqs|rope_theta|rope_scaling|Sinusoidal|pos_encoding|position_embedding|torch\\.arange|torch\\.full|output_hidden_states|\\.cpu\\(|\\.numpy\\(|\\.item\\(" "$(python - <<'PY'
import inspect, gr00t, pathlib
print(pathlib.Path(inspect.getfile(gr00t)).parents[1])
PY
)"
```

Profiler 里重点看这些事件：

```text
aten::arange
aten::full
aten::embedding
aten::cat
aten::contiguous
aten::copy_
aten::to
aten::_to_copy
aten::sin
aten::cos
aten::exp
aten::cpu
aten::item
aten::_local_scalar_dense
cudaMemcpyDtoH
cudaDeviceSynchronize
```

判据：

- 如果 RoPE/sin/cos/arange 相关事件每 request 合计不到 1 ms，先不要改。
- 如果 `normalized_action.cpu()` 附近暴露出大同步，要区分“copy 本身慢”还是“前面 GPU work 被同步计入”。
- 如果 action-head fixed tensor 构造在 8-step loop 中重复且可见，先做 request-local cache，要求固定 observation action diff 为 0。
- 如果 hidden states 存储或 backbone feature extraction 是主耗时，单独开一个 Phase 19 backbone contract 优化，不和 action-head 小修混在一起。

## 建议顺序

| rank | item | reason | gate |
| --- | --- | --- | --- |
| 1 | Qwen3 RoPE installed-source audit | 这是用户关心的 RoPE 位置，也是最容易看错版本的位置 | 确认实际 `transformers` 版本和源码路径 |
| 2 | `torch.profiler` sync/D2H scan | 先量化 `.cpu()`、`item()`、`cudaDeviceSynchronize` 是否真实存在 | 找到 >1 ms/request 的候选 |
| 3 | action-head fixed tensor cache | 低风险、固定 shape、容易 bit-exact | 固定 obs `max_abs_diff=0` |
| 4 | `SinusoidalPositionalEncoding` frequency buffer | 针对 `arange/log/exp/sin/cos` 重复构造 | profiler 证明收益，不改变数值 |
| 5 | DiT timestep/contiguous cache | 可能改善小长尾 | fixed replay bit-exact |
| 6 | backbone hidden-state contract | 潜在收益可能更大，但风险高 | 先固定 obs 证明 `last_hidden_state` 与旧特征一致 |

## 来源

- NVIDIA/Isaac-GR00T `n1.5-release` `eagle2_hg_model/config.json`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/backbone/eagle2_hg_model/config.json
- NVIDIA/Isaac-GR00T `n1.5-release` `modeling_eagle2_5_vl.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py
- NVIDIA/Isaac-GR00T `n1.5-release` `flow_matching_action_head.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/action_head/flow_matching_action_head.py
- NVIDIA/Isaac-GR00T `n1.5-release` `cross_attention_dit.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/action_head/cross_attention_dit.py
- NVIDIA/Isaac-GR00T `n1.5-release` `action_encoder.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/action_head/action_encoder.py
- NVIDIA/Isaac-GR00T `n1.5-release` `policy.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/policy.py
- NVIDIA/Isaac-GR00T `n1.5-release` `eagle_backbone.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/backbone/eagle_backbone.py
- NVIDIA/Isaac-GR00T `n1.5-release` `gr00t_n1.py`: https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/gr00t/model/gr00t_n1.py
