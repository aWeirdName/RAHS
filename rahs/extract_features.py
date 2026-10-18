"""Extract frozen CLIP/IP-Adapter sub-features without a diffusion pipeline."""

import argparse
from pathlib import Path

import torch

from .data import image_id
from .resampler import Resampler


def latent_importance(attention, num_tokens):
    """Original final-layer attention recipe: mean heads/queries, latent keys."""
    if attention.ndim != 4 or attention.shape[-1] < num_tokens:
        raise ValueError("Expected attention [B, heads, queries, image_keys + latent_keys]")
    return attention.float().mean(1).mean(1)[:, -num_tokens:]


def load_resampler(checkpoint, embedding_dim, device, dtype):
    """Load only image_proj weights from IP-Adapter Plus SDXL ViT-H."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint))
        state = {k.removeprefix("image_proj."): v for k, v in state.items() if k.startswith("image_proj.")}
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["image_proj"]
    tokens, dim = state["latents"].shape[1:]
    output_dim = state["proj_out.weight"].shape[0]
    # Checkpoint-compatible architecture used in the experimental implementation.
    if dim != 1280 or tokens != 16:
        raise ValueError("Expected IP-Adapter Plus SDXL ViT-H checkpoint (16 x 1280 latents)")
    model = Resampler(dim=dim, depth=4, dim_head=64, heads=20, num_queries=tokens,
                      embedding_dim=embedding_dim, output_dim=output_dim, ff_mult=4)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="training_images or test_images directory")
    parser.add_argument("--image-encoder", required=True, help="Local CLIP ViT-H encoder directory or Hugging Face model ID")
    parser.add_argument("--adapter-checkpoint", required=True, help="ip-adapter-plus_sdxl_vit-h.bin")
    parser.add_argument("--output", required=True, help="image_latent_train.pt or image_latent_test.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    # Optional dependencies are imported only when extracting real image features.
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    root = Path(args.images)
    paths = sorted(p for p in root.glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error(f"No images found in category subdirectories under {root}")
    ids = [image_id(p) for p in paths]
    if len(ids) != len(set(ids)):
        parser.error("Duplicate image IDs")
    dtype = torch.float16 if torch.device(args.device).type == "cuda" else torch.float32
    encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder).to(device=args.device, dtype=dtype)
    encoder.eval().requires_grad_(False)
    processor = CLIPImageProcessor()
    resampler = load_resampler(args.adapter_checkpoint, encoder.config.hidden_size, args.device, dtype)
    all_latents, all_importance = [], []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            images = []
            for path in paths[start:start + args.batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(device=args.device, dtype=dtype)
            hidden = encoder(pixels, output_hidden_states=True).hidden_states[-2]
            _, latents, attention = resampler(hidden)
            all_latents.append(latents.float().cpu())
            all_importance.append(latent_importance(attention, latents.shape[1]).cpu())
            print(f"{min(start + args.batch_size, len(paths))}/{len(paths)}", flush=True)
    importance = torch.cat(all_importance)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Latents": torch.cat(all_latents), "token_importance": importance,
                "topk_idx": importance.argsort(dim=1, descending=True), "image_ids": ids,
                "metadata": {"image_encoder": args.image_encoder,
                             "adapter_checkpoint": Path(args.adapter_checkpoint).name,
                             "importance": "final-layer mean over heads and queries; latent-key slice"}}, output)


if __name__ == "__main__":
    main()
