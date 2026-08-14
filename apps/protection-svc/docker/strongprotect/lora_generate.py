"""Coin-system feature (2026-08-10): train a real, downloadable SD1.5 LoRA
on a single one of the user's own artworks and save it as a `.safetensors`
file the user can actually use elsewhere -- unlike protection_score.py's
identically-shaped training (which trains a LoRA purely to CLIP-score it,
then throws the weights away), this keeps them.

Reuses protection_score.py's own `_train_lora_sd15` training loop (same
peft LoraConfig, same rank/alpha/LR, same single-image overfit-style
training) rather than a second copy -- the mechanism is identical, only
what happens to the result differs. v1 scope is deliberately SD1.5-only,
single source image: SDXL and multi-image datasets would need a real
dataset-loader rewrite of the training loop, out of scope for this
feature's first version (see PROJECT_DESIGN's coin-system plan doc).

Runs inside the same RunPod Serverless strong_protection worker as
protection_score.py (see docker/strongprotect-serverless/handler.py's
"generate_lora" action) -- same checkpoints, same ASPLAttacker class, no
separate image needed.

Progress goes to stderr; the CLI entrypoint's final
`print(json.dumps(result))` on stdout is the only thing a caller (the
handler, via subprocess) should parse -- same convention as
protection_score.py.
"""

import argparse
import json
import sys

import torch

from protection_score import _train_lora_sd15  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def generate_lora(
    image_path: str,
    output_path: str,
    prompt: str,
    sd15_checkpoint: str,
    seed: int = 1,
    train_steps: int = 150,
) -> dict:
    from aspl_attack import ASPLAttacker
    from caption_image import caption_image
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    # Same reasoning as protection_score.py's identical fix: the artwork's
    # own title is not a reliable training prompt, a real content caption
    # is.
    content_prompt = caption_image(image_path)
    _log(f"[lora_generate] caller-supplied prompt: {prompt!r} -- training with content caption instead: {content_prompt!r}")
    prompt = content_prompt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _log("[lora_generate] loading SD1.5 and training LoRA ...")
    sd15 = ASPLAttacker(sd15_checkpoint, device)
    text_embeddings = sd15.encode_prompt(prompt)

    lora_unet = _train_lora_sd15(sd15, image_path, text_embeddings, seed, train_steps)

    state_dict = get_peft_model_state_dict(lora_unet)
    # safetensors requires contiguous, detached, CPU tensors.
    state_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(state_dict, output_path)
    _log(f"[lora_generate] wrote {output_path}")

    lora_unet.unload()
    del sd15
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"outputPath": output_path, "contentPrompt": prompt, "trainSteps": train_steps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sd15-checkpoint", required=True)
    parser.add_argument("--prompt", default="artwork")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=150)
    args = parser.parse_args()

    result = generate_lora(
        image_path=args.image,
        output_path=args.output,
        prompt=args.prompt,
        sd15_checkpoint=args.sd15_checkpoint,
        seed=args.seed,
        train_steps=args.train_steps,
    )
    print(json.dumps(result))
