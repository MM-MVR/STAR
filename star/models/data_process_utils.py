import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

BACKGROUND_COLOR=(127, 127, 127)

def preprocess_image_with_min_size(image, min_factor=28):
    width, height = image.size 
    if height < min_factor or width < min_factor:
        scale_factor = max(min_factor / height, min_factor / width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)    
    return image

def preprocess_image_gen(images, processor, vq_transform):

    image_list = []
    grid_thw_list = []
    vq_image_list = []
    for image in images:
        image = preprocess_image_with_min_size(image)

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_list.append(image_tensor)
            
        grid_thw = visual_processed["image_grid_thw"][0]
        grid_thw_list.append(grid_thw)

        vq_image = vq_transform(image)
        vq_image_list.append(vq_image)
    
    image_tensor = torch.stack(image_list, dim=0)
    grid_thw = torch.stack(grid_thw_list, dim=0)
    vq_image = torch.stack(vq_image_list, dim=0)
        
    return {
        "pixel_values": image_tensor,
        "image_grid_thw": grid_thw,
        "vq_pixel_values": vq_image
    }



def get_vq_transform(args):
    return transforms.Compose([
        transforms.Resize((args.vq_image_size, args.vq_image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),    # [0, 255] -> [0, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),    # [0, 1] -> [-1, 1]
    ])

def get_full_transform(args):
    return transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),    # [0, 255] -> [0, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),    # [0, 1] -> [-1, 1]
    ])
