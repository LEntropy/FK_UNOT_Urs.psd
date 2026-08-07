"""Generate 10 diverse synthetic subculture character illustrations via
Illustrious-XL txt2img -- copyright-clean stand-ins for the n=30 SDXL-only
ASPL validation, matching the domain (anime-style original character art)
that DONTAI actually protects. The original 10-image set used for the
SD1.5+SDXL joint n=30 validation lived only on now-terminated pods and was
never saved locally, so this regenerates a fresh, comparably diverse set
rather than trying to recover the old one.

Run directly with python3 on a pod with the Illustrious-XL checkpoint.
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

PROMPTS = {
    "samurai_duel": "anime original character, samurai warrior mid-duel, katana, dynamic action pose, autumn leaves, detailed illustration",
    "silver_garden": "anime original character, girl in silver dress standing in a moonlit garden, fantasy, detailed illustration",
    "winter_scarf": "anime original character, boy wearing a red scarf in snowy street, winter coat, detailed illustration",
    "desert_wanderer": "anime original character, hooded wanderer walking through desert dunes, sunset, detailed illustration",
    "library_scholar": "anime original character, scholar surrounded by floating books in an old library, detailed illustration",
    "ice_skater": "anime original character, figure skater mid-spin on an ice rink, sparkling costume, detailed illustration",
    "punk_guitarist": "anime original character, punk rock guitarist on stage, spiked hair, neon stage lights, detailed illustration",
    "midnight_cafe": "anime original character, girl reading at a window seat in a midnight cafe, warm lighting, detailed illustration",
    "forest_ranger": "anime original character, forest ranger with a bow, standing in a misty pine forest, detailed illustration",
    "neon_arcade": "anime original character, gamer at a retro arcade cabinet, neon cyberpunk lighting, detailed illustration",
}

# Independent replication set for the n=30 SDXL-only ASPL validation -- a
# disjoint image set (no name/subject overlap with PROMPTS above) used to
# confirm the n=30 result isn't specific to that particular image set,
# mirroring the SD1.5 validation's own n=30 -> independent n=12 replication.
REPLICATION_PROMPTS = {
    "violin_prodigy": "anime original character, girl playing violin on a concert stage, spotlight, elegant dress, detailed illustration",
    "festival_lantern": "anime original character, boy holding a paper lantern at a night festival, fireworks, yukata, detailed illustration",
    "cyber_detective": "anime original character, detective in a trench coat investigating a neon-lit alley, cyberpunk noir, detailed illustration",
    "mountain_shrine": "anime original character, shrine maiden standing before a mountain torii gate, autumn mist, detailed illustration",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--set", default="original", choices=["original", "replication"])
    args = parser.parse_args()

    device = torch.device("cuda")
    pipe = StableDiffusionXLPipeline.from_single_file(args.checkpoint, torch_dtype=torch.float16, safety_checker=None).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS if args.set == "original" else REPLICATION_PROMPTS
    for i, (name, prompt) in enumerate(prompts.items()):
        out_path = out_dir / f"{name}.png"
        if out_path.exists():
            print(f"[{name}] already exists, skipping")
            continue
        generator = torch.Generator(device=device).manual_seed(args.seed + i)
        image = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.0, height=1024, width=1024, generator=generator).images[0]
        image.save(out_path)
        print(f"[{name}] wrote {out_path}")


if __name__ == "__main__":
    main()
