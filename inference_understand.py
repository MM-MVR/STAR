import os
import sys
import torch
import time
import random
import argparse
import numpy as np
from PIL import Image

from star.models.config import load_config_from_json
from star.models.config import STARMultiModalConfig
from star.models.model import STARMultiModal


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="Image understanding and question answering")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image or image URL")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")

    parser.add_argument("--data-type", type=str, default="understanding", help="Data type for model")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--diffusion-as-decoder", action="store_true", help="Whether to use DiT decoder")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--max-text-tokens", type=int, default=512, help="Max text tokens")
    parser.add_argument("--max-pixels", type=int, default=28 * 28 * 1024, help="Maximum number of pixels")
    parser.add_argument("--min-pixels", type=int, default=28 * 28 * 16, help="Minimum number of pixels")

    args = parser.parse_args()
    return args


def print_with_time(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: {msg}")


def model_setup(args, device):
    config_data = load_config_from_json(args.model_config)
    model_config = STARMultiModalConfig(**config_data)
    model = STARMultiModal(model_config, args)

    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    model.to(device).to(torch.bfloat16)
    return model


def load_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        print_with_time(f"Loaded image from local path: {image_path}")
    elif image_path.startswith(('http://', 'https://')):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes).convert('RGB')
            print_with_time(f"Loaded image from URL: {image_path}")
        else:
            raise ValueError(f"Failed to load image from URL {image_path}")
    else:
        raise ValueError(f"Invalid image path: {image_path}")

    print_with_time(f"Image size: {image.size}")
    return image


def understand_image(model, image, question, args):
    print_with_time(f"Question: {question}")

    with torch.no_grad():
        answer = model.inference_understand(
            image=image,
            question=question,
            max_new_tokens=args.max_new_tokens
        )

    return answer


def main():
    args = get_args()
    print(f"Arguments: {args}")

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print_with_time(f"Using device: {device}")

    print_with_time("Loading model...")
    model = model_setup(args, device)
    print_with_time("Model loaded successfully!")

    print_with_time("Loading image...")
    image = load_image(args.image_path)

    print_with_time("Starting image understanding...")
    answer = understand_image(model, image, args.question, args)

    print_with_time("=" * 50)
    print_with_time("RESULTS:")
    print_with_time(f"Image: {args.image_path}")
    print_with_time(f"Question: {args.question}")
    print_with_time(f"Answer: {answer}")
    print_with_time("=" * 50)

    print_with_time("Image understanding completed!")


if __name__ == '__main__':
    main()
