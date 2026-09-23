#!/usr/bin/env python3
"""Minimal ComprExIT demo built on the official repository APIs.

Run this file from the root of the cloned Jiangnan0522/ComprExIT repository.
It does two things on one QA example:
  1) runs the official compressed-context generation path;
  2) probes the learned depth-wise layer gates and width-wise OT plan.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from src.data_processing.prompt_template import QATemplate
from src.evaluation.utils_eval import get_model_and_tokenizer
from src.model.inference import compressing_predict_with_question_and_context


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def inspect_latent_message(model, tokenizer, context: str, device: str, max_context_length: int) -> None:
    """Inspect ComprExIT's token anchors / OT allocation for a single context."""
    if not hasattr(model, "pooling_layer"):
        print("[inspect] This checkpoint has no pooling_layer; skipping latent inspection.")
        return

    pooling = model.pooling_layer
    if not hasattr(pooling, "return_ot_plan") or not hasattr(pooling, "return_layer_weights"):
        print("[inspect] Pooling layer does not expose OT/layer diagnostics; skipping.")
        return

    old_padding_side = tokenizer.padding_side
    old_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        batch = tokenizer(
            [context],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_context_length,
        ).to(device)
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.truncation_side = old_truncation_side

    # The official model also moves context padding to the left before compression.
    context_ids, context_mask, _ = model.move_padding(
        batch.input_ids, batch.attention_mask, padding_side="left"
    )

    old_return_ot = pooling.return_ot_plan
    old_return_layers = pooling.return_layer_weights
    pooling.return_ot_plan = True
    pooling.return_layer_weights = True

    try:
        with torch.no_grad():
            outputs = pooling(
                context_ids,
                context_mask,
                backbone=model.compressor,
                tokenizer=tokenizer,
            )
            z_raw = outputs.pooled
            z_mask = outputs.pooled_mask
            z_decoder = model.projector(z_raw)
    finally:
        pooling.return_ot_plan = old_return_ot
        pooling.return_layer_weights = old_return_layers

    n_context = int(context_mask.sum().item())
    n_slots = int(z_mask.sum().item())
    effective_ratio = n_context / max(n_slots, 1)

    print("\n=== ComprExIT latent message ===")
    print(f"context tensor        : {tuple(context_ids.shape)}")
    print(f"raw compressed Z      : {tuple(z_raw.shape)}")
    print(f"decoder-space Z       : {tuple(z_decoder.shape)}")
    print(f"valid context tokens  : {n_context}")
    print(f"valid compression slots: {n_slots}")
    print(f"effective token ratio : {effective_ratio:.2f}x")

    # Depth-wise transmission: average the per-token layer gates over real context tokens.
    layer_weights = getattr(outputs, "layer_weights", None)
    if layer_weights is not None:
        valid = context_mask[0].bool()
        avg_layer = layer_weights[0, valid].float().mean(dim=0)
        k = min(6, avg_layer.numel())
        top = torch.topk(avg_layer, k=k)
        print("\n[depth-wise] mean gate weight, top layers")
        for layer_idx, weight in zip(top.indices.tolist(), top.values.tolist()):
            label = "embedding" if layer_idx == 0 else f"layer {layer_idx}"
            print(f"  {label:>10s}: {weight:.4f}")

    # Width-wise transmission: summing each OT row over slots recovers source mass.
    ot_plan = getattr(outputs, "ot_plan", None)
    if ot_plan is not None:
        # [B, n_windows, W, K] -> [B, n_windows * W]
        source_mass = ot_plan.sum(dim=-1).reshape(ot_plan.shape[0], -1)[0]
        source_mass = source_mass[: context_ids.shape[1]]
        valid = context_mask[0].bool()
        valid_positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        valid_mass = source_mass[valid_positions]
        top_k = min(10, valid_mass.numel())
        if top_k > 0:
            local_top = torch.topk(valid_mass, k=top_k)
            absolute_pos = valid_positions[local_top.indices]
            token_ids = context_ids[0, absolute_pos].tolist()
            token_strs = tokenizer.convert_ids_to_tokens(token_ids)
            print("\n[width-wise] highest learned source-mass tokens")
            for pos, tok, mass in zip(
                absolute_pos.tolist(), token_strs, local_top.values.tolist()
            ):
                print(f"  pos={pos:>3d}  mass={mass:.5f}  token={tok!r}")

        # Quick marginal check for the first non-empty window.
        window_mass = ot_plan.sum(dim=(-1, -2))[0]  # [n_windows]
        nonempty = torch.nonzero(window_mass > 0, as_tuple=False).squeeze(-1)
        if nonempty.numel() > 0:
            w = int(nonempty[0].item())
            p = ot_plan[0, w].float()
            print("\n[OT sanity check] first non-empty window")
            print(f"  window index        : {w}")
            print(f"  total transported mass: {p.sum().item():.6f}")
            print(f"  active source rows  : {(p.sum(-1) > 0).sum().item()}")
            print(f"  active target slots : {(p.sum(-2) > 0).sum().item()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal single-example ComprExIT demo")
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/comprexit-1b-4x-sft",
        help="Local path to the released ComprExIT SFT checkpoint.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--context",
        default=(
            "The Eiffel Tower is located in Paris, France. It was designed by engineer "
            "Gustave Eiffel's company and was completed in 1889 for the Exposition Universelle. "
            "The tower is one of the best-known landmarks in Paris."
        ),
    )
    parser.add_argument(
        "--question",
        default="In which city is the Eiffel Tower located?",
    )
    parser.add_argument("--no-inspect", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")

    model, tokenizer = get_model_and_tokenizer(
        args.checkpoint,
        device=device,
        attn_implementation="eager",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    template = QATemplate()
    formatted_context = template.build_context_text(args.context)
    formatted_question = template.build_question_text(args.question)

    if not args.no_inspect:
        inspect_latent_message(
            model,
            tokenizer,
            formatted_context,
            device,
            args.max_context_length,
        )

    print("\n=== Compressed QA generation ===")
    predictions = compressing_predict_with_question_and_context(
        questions=[formatted_question],
        contexts=[formatted_context],
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=template.prompt_text,
        answer_extractor=None,
        max_new_tokens=args.max_new_tokens,
        max_question_length=64,
        max_context_length=args.max_context_length,
        compress=True,
        with_context=True,
        compress_ratio=4,
    )

    print(f"Q: {args.question}")
    print(f"A: {predictions[0]}")


if __name__ == "__main__":
    main()
