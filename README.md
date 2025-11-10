<p align="center">
  <img src="assets/star_logo.png" alt="STAR" width="560"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img
      src="https://img.shields.io/badge/STAR-Paper-red?logo=arxiv&logoColor=red"
      alt="STAR Paper on arXiv"
    />
  </a>
  <a href="#">
    <img
      src="https://img.shields.io/badge/STAR-Project-0A66C2?logo=safari&logoColor=white"
      alt="STAR Project"
    />
  </a>
  <a href="#">
    <img 
        src="https://img.shields.io/badge/STAR-Models-yellow?logo=huggingface&logoColor=yellow" 
        alt="STAR Models"
    />
  </a>
  <a href="#">
    <img
      src="https://img.shields.io/badge/STAR-Demo-blue?logo=googleplay&logoColor=blue"
      alt="STAR Demo"
    />
  </a>
  <a href="#">
    <img 
        src="https://img.shields.io/badge/STAR-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="STAR HuggingFace Space"
    />
  </a>
</p>

# **STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning**


Welcome to the official repository for our paper: "STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning"


## **Abstract**
Multimodal large language models (MLLMs) play a pivotal role in advancing the quest for general artificial intelligence. However, achieving unified target for multimodal understanding and generation remains challenging due to optimization conflicts and performance trade-offs. To effectively enhance generative performance while preserving existing comprehension capabilities, we introduce ***STAR***: *a **ST**acked **A**uto**R**egressive scheme for task-progressive unified multimodal learning*. This approach decomposes multimodal learning into multiple stages: understanding, generation, and editing. By freezing the parameters of the fundamental autoregressive (AR) model and progressively stacking isomorphic AR modules, it avoids cross-task interference while expanding the model's capabilities. Concurrently, we introduce a high-capacity VQ to enhance the granularity of image representations and employ an implicit reasoning mechanism to improve generation quality under complex conditions. Experiments demonstrate that STAR achieves state-of-the-art performance on GenEval (**0.91**), DPG-Bench (**87.44**), and ImgEdit (**4.34**), validating its efficacy for unified multimodal learning.

<div align="center">
  <img src="assets/teaser.png" width=100%></img>
</div>

<div align="center">
  <img src="assets/STAR-Framework.png" width=100%></img>
</div>

## 🌟 Model Checkpoint


| Model Name | Checkpoint | Config |
| :--------: | :--------: | :----: |
| STAR-3B | [Link](#) | [Config](star/configs/STAR_Qwen2.5-VL-3B.json) |
| STAR-7B | [Link](#) | [Config](star/configs/STAR_Qwen2.5-VL-7B.json) |
| VQ Model | [Link](#) | - |


## 📚 Preparation

### Prepare the environment

1. Set up environment
```shell
git clone <repository-url>
cd STAR
conda create -n star python==3.11 -y
conda activate star
```

2. Install the required packages:
```shell
# upgrade pip and setuptools if necessary
pip install -U pip setuptools

# install required packages
pip install -r requirements.txt

```

### Download Pre-trained Models
Download the necessary pre-trained models before proceeding to inference.

```shell
STAR/checkpoints/STAR-7B.pt
STAR/checkpoints/VQ-Model.pt
```

### Configuration

The model configuration file `star/configs/STAR_Qwen2.5-VL-7B.json` contains all necessary parameters for model initialization. Make sure to update the paths in the configuration file to match your local setup.

## 🔥 Quick Start

### Demo

Run the interactive demo interface using Gradio.

```shell
python3 gradio_app.py 
```

### Inference

### 1. Image Understanding

For visual question answering and image understanding tasks:

```shell
python3 inference_understand.py \
    --image-path "path/to/your/image.jpg" \
    --question "What is in this image? Describe it in detail." \
    --max-new-tokens 256 \
    --model-config "star/configs/STAR_Qwen2.5-VL-7B.json" \
    --checkpoint "checkpoints/STAR-7B.pt" \
    --device "cuda:0"
```

**Parameters:**
- `--image-path`: Path to the input image
- `--question`: Question or instruction for the model
- `--max-new-tokens`: Maximum number of tokens to generate (default: 256)
- `--model-config`: Path to model configuration file
- `--checkpoint`: Path to model checkpoint
- `--device`: Device to run inference on

### 2. Text-to-Image Generation

For generating images from text prompts:

```shell
python3 inference_generation.py \
    --prompt "a photo of a cute cat" \
    --save-path "./outputs/a photo of a cute cat.jpg" \
    --num-images 1 \
    --cfg 1.1 \
    --topk 1000 \
    --topp 0.8 \
    --model-config "star/configs/STAR_Qwen2.5-VL-7B.json" \
    --checkpoint "checkpoints/STAR-7B.pt" \
    --diffusion-as-decoder \
    --device "cuda:0"
```

**Parameters:**
- `--prompt`: Text prompt for image generation
- `--save-path`: Path to save the generated image
- `--num-images`: Number of images to generate (default: 1)
- `--cfg`: Classifier-free guidance scale (default: 1.0)
- `--topk`: Top-k sampling parameter (default: 1000)
- `--topp`: Top-p sampling parameter (default: 0.8)
- `--diffusion-as-decoder`: Use diffusion model as decoder for high-quality generation

### 3. Image Editing

For editing images based on text instructions:

```shell
python3 inference_edit.py \
    --image-path "./outputs/a photo of a cute cat.jpg" \
    --instruction "change the color of cat to blue" \
    --save-path "./outputs/edited_image.jpg" \
    --cfg 1.1 \
    --topk 1000 \
    --topp 0.8 \
    --model-config "star/configs/STAR_Qwen2.5-VL-7B.json" \
    --checkpoint "checkpoints/STAR-7B.pt" \
    --diffusion-as-decoder \
    --device "cuda:0"
```

**Parameters:**
- `--image-path`: Path to the input image to be edited
- `--instruction`: Text instruction describing the desired edit
- `--save-path`: Path to save the edited image
- `--cfg`: Classifier-free guidance scale for editing
- `--topk`: Top-k sampling parameter
- `--topp`: Top-p sampling parameter
- `--diffusion-as-decoder`: Use diffusion model for high-quality image decoding



## 📊 Performance


### 1. Visual Understanding

| Model | #LLM | MMB | MMStar | MathVista | SEED | MME-P | MMMU | OCRBench | POPE | DocVQA |
| ----- | ---- | --- | ------ | --------- | ---- | ----- | ---- | -------- | ---- | ------ |
| Janus-Pro | 7B | 79.2 | 87.4 | - | 72.1 | 1567.1 | 41.0 | - | - | - |
| BLIP3-o | 8B | 83.5 | - | - | 77.5 | 1682.6 | 50.6 | - | - | - |
| Show-o2 | 7B | 79.3 | 56.6 | - | 69.8 | 1620.0 | 48.9 | - | - | - |
| MetaQuery-XL | 7B | 83.5 | - | - | 76.9 | 1685.2 | 58.6 | - | - | - |
| Bagel | 14B | 85.0 | - | 73.1 | - | 1687.0 | 55.3 | - | - | - |
| Ovis-U1 | 1.5B | 77.8 | - | 69.4 | - | - | 51.1 | 88.3 | - | - |
| ILLUME+ | 3B | 80.8 | - | - | 73.3 | 1414.0 | 44.3 | 67.2 | 87.6 | 80.8 |
| X-Omni | 7B | 74.8 | - | - | 74.1 | - | - | 70.4 | 89.3 | 88.6 |
| **STAR-3B** | 3B | **80.1** | **55.8** | **62.3** | **74.0** | **1592.3** | **53.1** | **79.7** | **85.9** | **93.9** |
| **STAR-7B** | 7B | **83.9** | **63.9** | **68.1** | **77.0** | **1690.1** | **58.6** | **86.4** | **86.6** | **95.7** |

### 2. Text-to-Image Generation

#### GenEval

| Model | Single | Two | Count. | Colors | Pos. | Color Attr. | Overall |
| ----- | ------ | --- | ------ | ------ | ---- | ----------- | ------- |
| **Generation-Only Models**|
| SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| DALL-E | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| SD3-medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
| FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| OmniGen2 | 0.99 | 0.96 | 0.74 | 0.98 | 0.72 | 0.75 | 0.86 |
| **Unified Models**</td> |
| Emu3 | 0.99 | 0.81 | 0.42 | 0.80 | 0.49 | 0.45 | 0.66 |
| ILLUME+ | 0.99 | 0.88 | 0.62 | 0.84 | 0.42 | 0.53 | 0.72 |
| Janus-Pro | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| MetaQuery | - | - | - | - | - | - | 0.80 |
| BLIP3-o | - | - | - | - | - | - | 0.84 |
| UniWorld-V1 | 0.99 | 0.93 | 0.81 | 0.89 | 0.74 | 0.71 | 0.84 |
| Mogao | 1.00 | 0.97 | 0.83 | 0.93 | 0.84 | 0.80 | 0.89 |
| BAGEL | 0.98 | 0.95 | 0.84 | 0.95 | 0.78 | 0.77 | 0.88 |
| Show-o2 | 1.00 | 0.87 | 0.58 | 0.92 | 0.52 | 0.62 | 0.76 |
| GPT-4o | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| X-Omni | 0.98 | 0.95 | 0.75 | 0.91 | 0.71 | 0.68 | 0.83 |
| Ovis-U1 | 0.98 | 0.98 | 0.90 | 0.92 | 0.79 | 0.75 | 0.89 |
| **STAR-3B** | 0.98 | 0.87 | 0.85 | 0.91 | 0.79 | 0.76 | **0.86** |
| **STAR-7B** | 0.98 | 0.94 | 0.90 | 0.92 | 0.91 | 0.80 | **0.91** |

#### DPG-Bench

| Model | Global | Entity | Attr. | Relation | Other | Overall |
| ----- | ------ | ------ | ----- | -------- | ----- | ------- |
| **Generation-Only Models** |
| SDXL | 83.27 | 82.43 | 80.91 | 86.76 | 80.41 | 74.65 |
| DALL-E | 90.97 | 89.61 | 88.39 | 90.58 | 89.83 | 83.50 |
| SD3-medium | 87.90 | 91.01 | 88.83 | 80.70 | 88.68 | 84.08 |
| FLUX.1-dev | 82.10 | 89.50 | 88.70 | 91.10 | 89.40 | 84.00 |
| OmniGen2 | 88.81 | 88.83 | 90.18 | 89.37 | 90.27 | 83.57 |
| **Unified Models** |
| Emu3 | 85.21 | 86.68 | 86.84 | 90.22 | 83.15 | 80.60 |
| ILLUME+ | - | - | - | - | - | - |
| Janus-Pro | 86.90 | 88.90 | 89.40 | 89.32 | 89.48 | 84.19 |
| MetaQuery | - | - | - | - | - | 82.05 |
| BLIP3-o | - | - | - | - | - | 81.60 |
| UniWorld-V1 | 83.64 | 88.39 | 88.44 | 89.27 | 87.22 | 81.38 |
| Mogao | 82.37 | 90.03 | 88.26 | 93.18 | 85.40 | 84.33 |
| BAGEL | 88.94 | 90.37 | 91.29 | 90.82 | 88.67 | 85.07 |
| Show-o2 | 89.00 | 91.78 | 89.96 | 91.81 | 91.64 | 86.14 |
| GPT-4o | 82.27 | 91.27 | 87.67 | 93.85 | 88.71 | 86.23 |
| X-Omni | 84.80 | 92.59 | 90.63 | 94.75 | 84.20 | 87.65 |
| Ovis-U1 | 82.37 | 90.08 | 88.68 | 93.35 | 85.20 | 83.72 |
| **STAR-3B** | 93.00 | 90.49 | 91.71 | 90.72 | 92.75 | **87.30** |
| **STAR-7B** | 94.97 | 92.91 | 91.62 | 94.30 | 83.82 | **87.44** |

#### WISE (World Knowledge Reasoning)

| Model | Cultural | Time | Space | Biology | Physics | Chemistry | Overall |
| ----- | -------- | ---- | ----- | ------- | ------- | --------- | ------- |
| **Generation-Only Models** |
| SD-XL | 0.43 | 0.48 | 0.47 | 0.44 | 0.45 | 0.27 | 0.43 |
| SD-3.5-large | 0.44 | 0.50 | 0.58 | 0.44 | 0.52 | 0.31 | 0.46 |
| FLUX.1-dev | 0.48 | 0.58 | 0.62 | 0.42 | 0.51 | 0.35 | 0.50 |
| **Unified Models** |
| Emu3 | 0.34 | 0.45 | 0.48 | 0.41 | 0.45 | 0.27 | 0.39 |
| Janus-Pro-7B | 0.30 | 0.37 | 0.49 | 0.36 | 0.42 | 0.26 | 0.35 |
| MetaQuery-XL | 0.56 | 0.55 | 0.62 | 0.49 | 0.63 | 0.41 | 0.55 |
| BLIP3-o | - | - | - | - | - | - | 0.62 |
| BAGEL | 0.76 | 0.69 | 0.75 | 0.65 | 0.75 | 0.58 | 0.70 |
| GPT-4o | 0.94 | 0.64 | 0.98 | 0.93 | 0.98 | 0.95 | 0.89 |
| **STAR-3B** | 0.58 | 0.54 | 0.48 | 0.49 | 0.51 | 0.54 | **0.52** |
| **STAR-7B** | 0.61 | 0.67 | 0.61 | 0.74 | 0.69 | 0.66 | **0.66** |

### 3. Image Editing

#### MagicBrush

| Model | L1 ↓ | CLIP-I ↑ | DINO ↑ |
| ----- | ---- | -------- | ------ |
| MagicBrush | 0.074 | 0.908 | 0.847 |
| Instruct-Pix2Pix | 0.114 | 0.851 | 0.744 |
| UltraEdit | 0.066 | 0.904 | 0.852 |
| ICEdit | 0.060 | 0.928 | 0.853 |
| OmniGen | 0.116 | 0.863 | 0.821 |
| UniReal | 0.081 | 0.903 | 0.837 |
| BAGEL | 0.074 | 0.914 | 0.827 |
| **STAR-3B** | **0.056** | **0.934** | **0.857** |
| **STAR-7B** | **0.060** | **0.931** | **0.853** |

#### ImgEdit-Bench

| Model | Add | Adjust | Extract | Replace | Remove | Background | Style | Hybrid | Action | Overall |
| ----- | --- | ------ | ------- | ------- | ------ | ---------- | ----- | ------ | ------ | ------- |
| **Editing-Only Models** |
| MagicBrush | 2.84 | 1.58 | 1.51 | 1.97 | 1.58 | 1.75 | 2.38 | 1.62 | 1.22 | 1.90 |
| Instruct-Pix2Pix | 2.45 | 1.83 | 1.44 | 2.01 | 1.50 | 1.44 | 3.55 | 1.20 | 1.46 | 1.88 |
| AnyEdit | 3.18 | 2.95 | 1.88 | 2.47 | 2.23 | 2.24 | 2.85 | 1.56 | 2.65 | 2.45 |
| UltraEdit | 3.44 | 2.81 | 2.13 | 2.96 | 1.45 | 2.83 | 3.76 | 1.91 | 2.98 | 2.70 |
| Step1X-Edit | 3.88 | 3.14 | 1.76 | 3.40 | 2.41 | 3.16 | 4.63 | 2.64 | 2.52 | 3.06 |
| ICEdit | 3.58 | 3.39 | 1.73 | 3.15 | 2.93 | 3.08 | 3.84 | 2.04 | 3.68 | 3.05 |
| **Unified Models** |
| GPT-4o | 4.61 | 4.33 | 2.90 | 4.35 | 3.66 | 4.57 | 4.93 | 3.96 | 4.89 | 4.20 |
| OmniGen | 3.47 | 3.04 | 1.71 | 2.94 | 2.43 | 3.21 | 4.19 | 2.24 | 3.38 | 2.96 |
| BAGEL | 3.56 | 3.31 | 1.70 | 3.30 | 2.62 | 3.24 | 4.49 | 2.38 | 4.17 | 3.20 |
| UniWorld-V1 | 3.82 | 3.64 | 2.27 | 3.47 | 3.24 | 2.99 | 4.21 | 2.96 | 2.74 | 3.26 |
| OmniGen2 | 3.57 | 3.06 | 1.77 | 3.74 | 3.20 | 3.57 | 4.81 | 2.52 | 4.68 | 3.44 |
| Ovis-U1 | 4.13 | 3.62 | 2.98 | 4.45 | 4.06 | 4.22 | 4.69 | 3.45 | 4.61 | 4.00 |
| **STAR-3B** | **4.26** | **4.06** | **3.78** | **4.46** | **4.34** | **4.19** | **4.53** | **3.29** | **4.38** | **4.14** |
| **STAR-7B** | **4.33** | **4.19** | **4.19** | **4.59** | **4.58** | **4.36** | **4.59** | **3.67** | **4.60** | **4.34** |


## ✍️ Citation

```bibtex
@article{2025star,
  title   = {STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning},
  author  = {Qin, Jie and Huang, Jiancheng and Qiao, Limeng and Ma, Lin},
  journal = {arXiv preprint arXiv:},
  year    = {2025}
}
```


## 📜 License
STAR is licensed under the Apache 2.0.
