"""Phase 6 W4A16 module-scope helpers.

These scopes are intentionally finer than the Phase 5 `config_groups` choices.
They let us evaluate deployment tradeoffs without changing earlier fake-quant
experiment semantics.
"""

from __future__ import annotations

from phase3_activation_capture import group_for_module


SCOPE_CHOICES = (
    "dit_mlp_only",
    "llm_attn_only",
    "llm_mlp_only",
    "llm_selected",
    "llm_mlp_dit_mlp",
    "llm_dit_mlp",
)


def module_family(name: str) -> str | None:
    group = group_for_module(name)
    if group == "dit_mlp_selected":
        return "dit_mlp"
    if group == "llm_selected" and ".self_attn." in name:
        return "llm_attn"
    if group == "llm_selected" and ".mlp." in name:
        return "llm_mlp"
    return None


def include_module_for_scope(name: str, scope: str) -> bool:
    family = module_family(name)
    if scope == "dit_mlp_only":
        return family == "dit_mlp"
    if scope == "llm_attn_only":
        return family == "llm_attn"
    if scope == "llm_mlp_only":
        return family == "llm_mlp"
    if scope == "llm_selected":
        return family in {"llm_attn", "llm_mlp"}
    if scope == "llm_mlp_dit_mlp":
        return family in {"llm_mlp", "dit_mlp"}
    if scope == "llm_dit_mlp":
        return family in {"llm_attn", "llm_mlp", "dit_mlp"}
    raise ValueError(f"Unknown Phase 6 W4A16 scope: {scope!r}. Choices: {', '.join(SCOPE_CHOICES)}")


def scope_description(scope: str) -> str:
    if scope == "dit_mlp_only":
        return "DiT feed-forward Linear modules only"
    if scope == "llm_attn_only":
        return "LLM self-attention Linear modules only"
    if scope == "llm_mlp_only":
        return "LLM MLP Linear modules only"
    if scope == "llm_selected":
        return "All selected LLM Linear modules: attention plus MLP"
    if scope == "llm_mlp_dit_mlp":
        return "LLM MLP plus DiT feed-forward Linear modules, excluding LLM attention"
    if scope == "llm_dit_mlp":
        return "Phase 5-compatible full selected scope: LLM attention, LLM MLP, and DiT MLP"
    raise ValueError(f"Unknown Phase 6 W4A16 scope: {scope!r}")

