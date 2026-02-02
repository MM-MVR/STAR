import os
import sys
import pickle
import torch
import time
import random
import argparse
import numpy as np
import json
from PIL import Image

from star.models.config import load_config_from_json
from star.models.config import STARMultiModalConfig
from star.models.model import STARMultiModal


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="Single image editing")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--instruction", type=str, required=True, help="Editing instruction")
    parser.add_argument("--save-path", type=str, default="./edited_image.jpg", help="Path to save edited image")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference")

    parser.add_argument("--data-type", type=str, default="img2img", help="Data type for model")
    parser.add_argument("--topk", type=int, default=2000)
    parser.add_argument("--cfg", type=float, default=20.0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--vq-image-size", type=int, default=384, help="Size of generated images")
    parser.add_argument("--vq-tokens", type=int, default=576, help="Number of VQ tokens")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")

    parser.add_argument("--diffusion-as-decoder", action="store_true", help="Whether to use DiT decoder")
    parser.add_argument("--ori-inp-dit", type=str, default="seq", help="Original input dit configuration")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--diffusion-resolution", type=int, default=1024, help="Diffusion resolution")
    parser.add_argument("--max-diff-seq-length", type=int, default=256, help="Maximum diffusion sequence length")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="maximum sequence length")
    parser.add_argument("--max-text-tokens", type=int, default=512, help="max text tokens")
    parser.add_argument("--max-pixels", type=int, default=28 * 28 * 576, help="maximum number of pixels")
    parser.add_argument("--min-pixels", type=int, default=28 * 28 * 16, help="minimum number of pixels")

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


def edit_single_image(model, image_path, instruction, args):
    image_size = args.vq_image_size
    num_return_sequences = args.num_images

    try:
        img_ori = Image.open(image_path).convert('RGB')
        print_with_time(f"Loaded image from: {image_path}")
    except Exception as e:
        print_with_time(f"Error loading image: {e}")
        return None

    print_with_time(f"Editing image with instruction: {instruction}")

    with torch.no_grad():
        output = model.generate_images_edit(
            [img_ori], 
            instruction,
            max_new_tokens=args.vq_tokens,
            num_return_sequences=num_return_sequences,
            cfg_weight=args.cfg,
            topk_sample=args.topk,
            topp_sample=args.topp,
            return_dict=True
        )

        if output is None:
            print("Editing failed!")
            return None

        if isinstance(output, dict):
            
            output_images = output.get("output_images")
            diff_images = output.get("diff_images")

            results = {}
            if output_images is not None:
                dec_vq = np.clip((output_images + 1) / 2 * 255, 0, 255)
                visual_img_vq = np.zeros((num_return_sequences, image_size, image_size, 3), dtype=np.uint8)
                visual_img_vq[:, :, :] = dec_vq
                imgs_vq = [Image.fromarray(visual_img_vq[j].astype(np.uint8)) for j in range(visual_img_vq.shape[0])]
                results["vq_images"] = imgs_vq

            if diff_images is not None:
                results["diff_images"] = diff_images
            else:
                results["diff_images"] = None

            return results
        else:
            dec = np.clip((output + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((num_return_sequences, image_size, image_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            imgs = [Image.fromarray(visual_img[j].astype(np.uint8)) for j in range(visual_img.shape[0])]
            return {"vq_images": imgs, "diff_images": None}


def save_images(imgs, save_path, instruction):
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if isinstance(imgs, dict):
        vq_images = imgs.get("vq_images")
        diff_images = imgs.get("diff_images")

        if vq_images is not None and len(vq_images) > 0:
            base_name, ext = os.path.splitext(save_path)
            vq_save_path = f"{base_name}_vq{ext}"
            vq_images[0].save(vq_save_path)
            print_with_time(f"VQ edited image saved to: {vq_save_path}")

        if diff_images is not None and len(diff_images) > 0:
            base_name, ext = os.path.splitext(save_path)
            diff_save_path = f"{base_name}_diff{ext}"
            diff_images[0].save(diff_save_path)
            print_with_time(f"Diff edited image saved to: {diff_save_path}")
    else:
        if isinstance(imgs, list) and len(imgs) > 0:
            imgs[0].save(save_path)
            print_with_time(f"Edited image saved to: {save_path}")
        else:
            print_with_time("No images to save!")


def main():
    args = get_args()
    print(f"Arguments: {args}")

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print_with_time(f"Using device: {device}")

    print_with_time("Loading model...")
    model = model_setup(args, device)
    print_with_time("Model loaded successfully!")

    print_with_time("Starting image editing...")
    imgs = edit_single_image(model, args.image_path, args.instruction, args)

    if imgs is not None:
        save_images(imgs, args.save_path, args.instruction)
        print_with_time("Image editing completed!")
    else:
        print_with_time("Image editing failed!")


if __name__ == '__main__':
    main()
