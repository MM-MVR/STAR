import os
import sys
import gradio as gr
import numpy as np
import torch
import random
import time
from PIL import Image

from star.models.config import load_config_from_json, STARMultiModalConfig
from star.models.model import STARMultiModal


TEXTS = {
    "zh": {
        "title": "🌟 STAR 多模态演示",
        "description": "基于STAR模型的多模态AI演示系统，支持文本生成图像、图像编辑和图像理解功能。",
        "please_load_model": "请先加载模型！",
        "please_upload_image": "请上传图像！",
        "generation_failed": "生成失败！",
        "generation_success_diffusion": "生成成功！",
        "generation_success_vq": "生成成功！",
        "edit_failed": "编辑失败！",
        "edit_success_diffusion": "编辑成功！",
        "edit_success_vq": "编辑成功！",
        "understanding_failed": "理解失败！",
        "generation_error": "生成过程中出错: ",
        "edit_error": "编辑过程中出错: ",
        "understanding_error": "理解过程中出错: ",
        "tab_text_to_image": "🖼️ 文本生成图像",
        "tab_image_edit": "🖌️ 图像编辑",
        "tab_image_understanding": "📝 图像理解",
        "text_prompt": "文本提示",
        "text_prompt_placeholder": "A whimsical scene featuring a small elf with pointed ears and a green hat, sipping orange juice through a long straw from a disproportionately large orange. Next to the elf, a curious squirrel perches on its hind legs, while an owl with wide, observant eyes watches intently from a branch overhead. The orange's vibrant color contrasts with the muted browns and greens of the surrounding forest foliage.",
        "advanced_params": "高级参数",
        "cfg_scale": "CFG Scale",
        "cfg_scale_info": "控制生成图像与文本的匹配程度",
        "top_k": "Top-K",
        "top_k_info": "采样时考虑的token数量",
        "top_p": "Top-P",
        "top_p_info": "核采样参数",
        "generate_image": "🎨 生成图像",
        "generated_image": "生成的图像",
        "generation_status": "生成状态",
        "input_image": "输入图像",
        "edit_instruction": "编辑指令",
        "edit_instruction_placeholder": "Remove the tiger in the water.",
        "edit_image": "✏️ 编辑图像",
        "edited_image": "编辑后的图像",
        "edit_status": "编辑状态",
        "question": "问题",
        "question_placeholder": "Please describe the content of this image",
        "max_generation_length": "最大生成长度",
        "understand_image": "🔍 理解图像",
        "understanding_result": "理解结果",
        "usage_instructions": "使用说明",
        "usage_step1": "1. **文本生成图像**: 输入文本描述，调整参数后点击生成",
        "usage_step2": "2. **图像编辑**: 上传图像并输入编辑指令",
        "usage_step3": "3. **图像理解**: 上传图像并提出问题",
        "language": "语言 / Language"
    },
    "en": {
        "title": "🌟 STAR Multi-Modal Demo",
        "description": "A multi-modal AI demonstration system based on STAR model, supporting text-to-image generation, image editing, and image understanding.",
        "please_load_model": "Please load the model first!",
        "please_upload_image": "Please upload an image!",
        "generation_failed": "Generation failed!",
        "generation_success_diffusion": "Generation successful! ",
        "generation_success_vq": "Generation successful! Using VQ decoder",
        "edit_failed": "Editing failed!",
        "edit_success_diffusion": "Editing successful! ",
        "edit_success_vq": "Editing successful! Using VQ decoder",
        "understanding_failed": "Understanding failed!",
        "generation_error": "Error during generation: ",
        "edit_error": "Error during editing: ",
        "understanding_error": "Error during understanding: ",
        "tab_text_to_image": "🖼️ Text to Image",
        "tab_image_edit": "🖌️ Image Editing",
        "tab_image_understanding": "📝 Image Understanding",
        "text_prompt": "Text Prompt",
        "text_prompt_placeholder": "A whimsical scene featuring a small elf with pointed ears and a green hat, sipping orange juice through a long straw from a disproportionately large orange. Next to the elf, a curious squirrel perches on its hind legs, while an owl with wide, observant eyes watches intently from a branch overhead. The orange's vibrant color contrasts with the muted browns and greens of the surrounding forest foliage.",
        "advanced_params": "Advanced Parameters",
        "cfg_scale": "CFG Scale",
        "cfg_scale_info": "Controls how closely the generated image matches the text",
        "top_k": "Top-K",
        "top_k_info": "Number of tokens to consider during sampling",
        "top_p": "Top-P",
        "top_p_info": "Nucleus sampling parameter",
        "generate_image": "🎨 Generate Image",
        "generated_image": "Generated Image",
        "generation_status": "Generation Status",
        "input_image": "Input Image",
        "edit_instruction": "Edit Instruction",
        "edit_instruction_placeholder": "Remove the tiger in the water.",
        "edit_image": "✏️ Edit Image",
        "edited_image": "Edited Image",
        "edit_status": "Edit Status",
        "question": "Question",
        "question_placeholder": "Please describe the content of this image",
        "max_generation_length": "Max Generation Length",
        "understand_image": "🔍 Understand Image",
        "understanding_result": "Understanding Result",
        "usage_instructions": "Usage Instructions",
        "usage_step1": "1. **Text to Image**: Enter text description, adjust parameters and click generate",
        "usage_step2": "2. **Image Editing**: Upload an image and enter editing instructions",
        "usage_step3": "3. **Image Understanding**: Upload an image and ask questions",
        "language": "语言 / Language"
    }
}


def set_seed(seed=100):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def print_with_time(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: {msg}")


class STARInferencer:

    def __init__(self, model_config_path, checkpoint_path, device="cuda:0"):
        self.device = device
        self.model_config_path = model_config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()

    def _create_mock_args(self):
        class MockArgs:
            def __init__(self):
                self.data_type = "generation"
                self.diffusion_as_decoder = True
                self.ori_inp_dit = "seq"
                self.grad_ckpt = False
                self.diffusion_resolution = 1024
                self.max_diff_seq_length = 256
                self.max_seq_length = 8192
                self.max_text_tokens = 512
                self.max_pixels = 28 * 28 * 576
                self.min_pixels = 28 * 28 * 16
                self.vq_image_size = 384
                self.vq_tokens = 576

        return MockArgs()

    def _load_model(self):
        try:
            print_with_time("Loading model configuration...")
            config_data = load_config_from_json(self.model_config_path)
            model_config = STARMultiModalConfig(**config_data)

            args = self._create_mock_args()

            print_with_time("Initializing model...")
            self.model = STARMultiModal(model_config, args)

            if os.path.exists(self.checkpoint_path):
                print_with_time(f"Loading checkpoint from {self.checkpoint_path}")
                with torch.no_grad():
                    checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    if not isinstance(state_dict, dict):
                        raise ValueError("Invalid checkpoint format")

                    print_with_time(f"Checkpoint contains {len(state_dict)} parameters")
                    self.model.load_state_dict(state_dict, strict=False)

            print_with_time(f"Moving model to device: {self.device}")
            self.model.to(self.device)

            print_with_time("Converting model to bfloat16...")
            self.model.to(torch.bfloat16)

            print_with_time("Setting model to eval mode...")
            self.model.eval()

            if torch.cuda.is_available():
                print_with_time(f"GPU memory after model loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            print_with_time("Model loaded successfully!")

        except Exception as e:
            print_with_time(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def generate_image(self, prompt, num_images=1, cfg=20.0, topk=2000, topp=1.0, seed=0):
        set_seed(seed)

        print_with_time(f"Generating image for prompt: {prompt}")

        cfg = max(1.0, min(20.0, float(cfg)))
        topk = max(100, min(2000, int(topk)))
        topp = max(0.1, min(1.0, float(topp)))

        print_with_time(f"Using validated params: cfg={cfg}, topk={topk}, topp={topp}")

        if not (torch.isfinite(torch.tensor(cfg)) and torch.isfinite(torch.tensor(topk)) and torch.isfinite(torch.tensor(topp))):
            print_with_time("Warning: Non-finite parameters detected")
            return None

        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print_with_time(f"GPU memory before generation: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

                if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                    print_with_time("Warning: Invalid prompt")
                    return None

                if not (0 < cfg <= 20 and 0 < topk <= 5000 and 0 < topp <= 1):
                    print_with_time(f"Warning: Invalid parameters - cfg={cfg}, topk={topk}, topp={topp}")
                    return None

                print_with_time("Calling model.generate_images...")

                safe_max_tokens = 576 

                output = self.model.generate_images(
                    prompt,
                    max_new_tokens=safe_max_tokens,
                    num_return_sequences=num_images,
                    cfg_weight=cfg,
                    topk_sample=topk,
                    topp_sample=topp,
                    reasoning=False,
                    return_dict=True
                )
                print_with_time("Model generation completed")

                if output is None:
                    print_with_time("Warning: Model returned None output")
                    return None

                print_with_time("Processing output images...")
                result = self._process_output_images(output, num_images)
                print_with_time("Image processing completed")
                return result
        except Exception as e:
            print_with_time(f"Error during image generation: {str(e)}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def edit_image(self, image, instruction, num_images=1, cfg=20.0, topk=2000, topp=1.0, seed=0):
        set_seed(seed)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        print_with_time(f"Editing image with instruction: {instruction}")

        with torch.no_grad():
            output = self.model.generate_images_edit(
                [image],
                instruction,
                max_new_tokens=576,
                num_return_sequences=num_images,
                cfg_weight=cfg,
                topk_sample=topk,
                topp_sample=topp,
                return_dict=True
            )

            if output is None:
                return None

            return self._process_output_images(output, num_images)

    def understand_image(self, image, question, max_new_tokens=256):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        print_with_time(f"Understanding image with question: {question}")

        with torch.no_grad():
            answer = self.model.inference_understand(
                image=image,
                question=question,
                max_new_tokens=max_new_tokens
            )

        return answer

    def _process_output_images(self, output, num_images):
        image_size = 384  

        try:
            if isinstance(output, dict):
                output_images = output.get("output_images")
                diff_images = output.get("diff_images")

                results = {}

                if output_images is not None:
                    if isinstance(output_images, torch.Tensor):
                        output_images = output_images.detach().cpu().numpy()

                    if output_images.size == 0:
                        print_with_time("Warning: Empty output_images array")
                        results["vq_images"] = None
                    else:
                        output_images = np.nan_to_num(output_images, nan=0.0, posinf=1.0, neginf=-1.0)
                        dec_vq = np.clip((output_images + 1) / 2 * 255, 0, 255)

                        if len(dec_vq.shape) == 3:
                            dec_vq = dec_vq.reshape(num_images, image_size, image_size, 3)

                        visual_img_vq = np.zeros((num_images, image_size, image_size, 3), dtype=np.uint8)
                        visual_img_vq[:, :, :] = dec_vq
                        imgs_vq = [Image.fromarray(visual_img_vq[j].astype(np.uint8)) for j in range(visual_img_vq.shape[0])]
                        results["vq_images"] = imgs_vq

                if diff_images is not None:
                    results["diff_images"] = diff_images
                else:
                    results["diff_images"] = None

                return results
            else:
                if isinstance(output, torch.Tensor):
                    output = output.detach().cpu().numpy()

                output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                dec = np.clip((output + 1) / 2 * 255, 0, 255)

                if len(dec.shape) == 3:
                    dec = dec.reshape(num_images, image_size, image_size, 3)

                visual_img = np.zeros((num_images, image_size, image_size, 3), dtype=np.uint8)
                visual_img[:, :, :] = dec
                imgs = [Image.fromarray(visual_img[j].astype(np.uint8)) for j in range(visual_img.shape[0])]
                return {"vq_images": imgs, "diff_images": None}

        except Exception as e:
            print_with_time(f"Error in _process_output_images: {str(e)}")
            return {"vq_images": None, "diff_images": None}


inferencer = None



def save_language_setting(language):
    try:
        with open('.language_setting', 'w') as f:
            f.write(language)
    except:
        pass


current_language = "en"  

def get_text(key):
    return TEXTS[current_language].get(key, key)


def auto_detect_device():
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        print_with_time(f"Detected CUDA device: {device}")
        print_with_time(f"GPU name: {torch.cuda.get_device_name()}")
        print_with_time(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        device = "cpu"
        print_with_time("No CUDA device detected, using CPU")
    return device


def initialize_model_on_startup():
    global inferencer

    default_config = "star/configs/STAR_Qwen2.5-VL-7B.json"
    default_checkpoint = "checkpoints/STAR-7B.pt"

    if not os.path.exists(default_config):
        print_with_time(f"⚠️ Model config file not found: {default_config}")
        return False, f"Model config file not found: {default_config}"

    if not os.path.exists(default_checkpoint):
        print_with_time(f"⚠️ Model checkpoint file not found: {default_checkpoint}")
        return False, f"Model checkpoint file not found: {default_checkpoint}"

    try:
        device = auto_detect_device()
        print_with_time("Starting to load STAR model...")

        inferencer = STARInferencer(default_config, default_checkpoint, device)

        print_with_time("✅ STAR model loaded successfully!")
        return True, "✅ STAR model loaded successfully!"

    except Exception as e:
        error_msg = f"❌ Model loading failed: {str(e)}"
        print_with_time(error_msg)
        return False, error_msg




def text_to_image(prompt, cfg_scale=1.0, topk=1000, topp=0.8):
    if inferencer is None:
        return None, get_text("please_load_model")

    cfg_scale = max(1.0, min(20.0, cfg_scale))  
    topk = max(100, min(2000, int(topk)))       
    topp = max(0.1, min(1.0, topp))             
    seed = 100

    try:
        print_with_time(f"Starting generation with params: cfg={cfg_scale}, topk={topk}, topp={topp}, seed={seed}")
        result = inferencer.generate_image(prompt, cfg=cfg_scale, topk=topk, topp=topp, seed=seed)

        if result is None:
            return None, get_text("generation_failed")

        if result.get("diff_images") and len(result["diff_images"]) > 0:
            return result["diff_images"][0], get_text("generation_success_diffusion")
        elif result.get("vq_images") and len(result["vq_images"]) > 0:
            return result["vq_images"][0], get_text("generation_success_vq")
        else:
            return None, get_text("generation_failed")

    except Exception as e:
        return None, get_text("generation_error") + str(e)


def image_editing(image, instruction, cfg_scale=1.0, topk=1000, topp=0.8):
    if inferencer is None:
        return None, get_text("please_load_model")

    if image is None:
        return None, get_text("please_upload_image")


    cfg_scale = max(1.0, min(20.0, cfg_scale))  
    topk = max(100, min(2000, int(topk)))       
    topp = max(0.1, min(1.0, topp))             
    seed = 100  

    try:
        print_with_time(f"Starting image editing with params: cfg={cfg_scale}, topk={topk}, topp={topp}, seed={seed}")
        result = inferencer.edit_image(image, instruction, cfg=cfg_scale, topk=topk, topp=topp, seed=seed)

        if result is None:
            return None, get_text("edit_failed")

        if result.get("diff_images") and len(result["diff_images"]) > 0:
            return result["diff_images"][0], get_text("edit_success_diffusion")
        elif result.get("vq_images") and len(result["vq_images"]) > 0:
            return result["vq_images"][0], get_text("edit_success_vq")
        else:
            return None, get_text("edit_failed")

    except Exception as e:
        return None, get_text("edit_error") + str(e)


def image_understanding(image, question, max_new_tokens=256):
    if inferencer is None:
        return get_text("please_load_model")

    if image is None:
        return get_text("please_upload_image")

    try:
        answer = inferencer.understand_image(image, question, max_new_tokens)
        return answer if answer else get_text("understanding_failed")

    except Exception as e:
        return get_text("understanding_error") + str(e)


def change_language(language):
    global current_language
    current_language = language

    return (
        get_text("title"),
        get_text("description"),
        get_text("tab_text_to_image"),
        get_text("text_prompt"),
        get_text("text_prompt_placeholder"),
        get_text("advanced_params"),
        get_text("cfg_scale"),
        get_text("cfg_scale_info"),
        get_text("top_k"),
        get_text("top_k_info"),
        get_text("top_p"),
        get_text("top_p_info"),
        get_text("random_seed"),
        get_text("random_seed_info"),
        get_text("generate_image"),
        get_text("generated_image"),
        get_text("generation_status"),
        get_text("tab_image_edit"),
        get_text("input_image"),
        get_text("edit_instruction"),
        get_text("edit_instruction_placeholder"),
        get_text("edit_image"),
        get_text("edited_image"),
        get_text("edit_status"),
        get_text("tab_image_understanding"),
        get_text("question"),
        get_text("question_placeholder"),
        get_text("max_generation_length"),
        get_text("understand_image"),
        get_text("understanding_result"),
        get_text("usage_instructions"),
        get_text("usage_step1"),
        get_text("usage_step2"),
        get_text("usage_step3")
    )


def load_example_image(image_path):
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
    return None



def create_interface():
    
    print_with_time("Initializing STAR demo system...")
    model_loaded, status_message = initialize_model_on_startup()

    with gr.Blocks(title="🌟 STAR Multi-Modal Demo", theme=gr.themes.Soft()) as demo:
        
        language_state = gr.State(value=current_language)
        title_md = gr.Markdown(f"# {get_text('title')}")
        desc_md = gr.Markdown(get_text("description"))

        with gr.Row():
            with gr.Column():
                language_dropdown = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh")],
                    value=current_language,
                    label="Language / 语言",
                    interactive=True
                )

        with gr.Tabs():
            with gr.Tab(get_text("tab_text_to_image")) as txt_tab:
                with gr.Row():
                    with gr.Column():
                        txt_prompt = gr.Textbox(
                            label=get_text("text_prompt"),
                            value=get_text("text_prompt_placeholder"),
                            lines=3
                        )

                        with gr.Accordion(get_text("advanced_params"), open=False):
                            txt_cfg_scale = gr.Slider(
                                minimum=1.0, maximum=20.0, value=1.1, step=0.1,
                                label=get_text("cfg_scale"), info=get_text("cfg_scale_info")
                            )
                            txt_topk = gr.Slider(
                                minimum=100, maximum=2000, value=1000, step=50,
                                label=get_text("top_k"), info=get_text("top_k_info")
                            )
                            txt_topp = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.8, step=0.05,
                                label=get_text("top_p"), info=get_text("top_p_info")
                            )

                        txt_generate_btn = gr.Button(get_text("generate_image"), variant="primary")

                    with gr.Column():
                        txt_output_image = gr.Image(label=get_text("generated_image"))
                        txt_status = gr.Textbox(label=get_text("generation_status"), interactive=False)

            
            with gr.Tab(get_text("tab_image_edit")) as edit_tab:
                with gr.Row():
                    with gr.Column():
                        edit_input_image = gr.Image(
                            label=get_text("input_image"),
                            value=load_example_image('assets/editing.png')
                        )
                        edit_instruction = gr.Textbox(
                            label=get_text("edit_instruction"),
                            value=get_text("edit_instruction_placeholder"),
                            lines=2
                        )

                        with gr.Accordion(get_text("advanced_params"), open=False):
                            edit_cfg_scale = gr.Slider(
                                minimum=1.0, maximum=20.0, value=1.1, step=0.1,
                                label=get_text("cfg_scale")
                            )
                            edit_topk = gr.Slider(
                                minimum=100, maximum=2000, value=1000, step=50,
                                label=get_text("top_k")
                            )
                            edit_topp = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.8, step=0.05,
                                label=get_text("top_p")
                            )

                        edit_btn = gr.Button(get_text("edit_image"), variant="primary")

                    with gr.Column():
                        edit_output_image = gr.Image(label=get_text("edited_image"))
                        edit_status = gr.Textbox(label=get_text("edit_status"), interactive=False)

            
            with gr.Tab(get_text("tab_image_understanding")) as understand_tab:
                with gr.Row():
                    with gr.Column():
                        understand_input_image = gr.Image(
                            label=get_text("input_image"),
                            value=load_example_image('assets/understand.png')
                        )
                        understand_question = gr.Textbox(
                            label=get_text("question"),
                            value=get_text("question_placeholder"),
                            lines=2
                        )

                        with gr.Accordion(get_text("advanced_params"), open=False):
                            understand_max_tokens = gr.Slider(
                                minimum=64, maximum=1024, value=256, step=64,
                                label=get_text("max_generation_length")
                            )

                        understand_btn = gr.Button(get_text("understand_image"), variant="primary")

                    with gr.Column():
                        understand_output = gr.Textbox(
                            label=get_text("understanding_result"),
                            lines=15,
                            interactive=False
                        )

        usage_md = gr.Markdown(
            f"""
            ---
            ### {get_text("usage_instructions")}
            {get_text("usage_step1")}
            {get_text("usage_step2")}
            {get_text("usage_step3")}
            """
        )

        txt_generate_btn.click(
            fn=text_to_image,
            inputs=[txt_prompt, txt_cfg_scale, txt_topk, txt_topp],
            outputs=[txt_output_image, txt_status]
        )

        edit_btn.click(
            fn=image_editing,
            inputs=[edit_input_image, edit_instruction, edit_cfg_scale, edit_topk, edit_topp],
            outputs=[edit_output_image, edit_status]
        )

        understand_btn.click(
            fn=image_understanding,
            inputs=[understand_input_image, understand_question, understand_max_tokens],
            outputs=understand_output
        )

        
        def update_interface_language(language):
            global current_language
            current_language = language

            save_language_setting(language)

            return [
                language,  
                f"# {get_text('title')}",  
                get_text("description"),  
                get_text("text_prompt_placeholder"),  
                get_text("edit_instruction_placeholder"),  
                get_text("question_placeholder"),  
                f"""
                ---
                ### {get_text("usage_instructions")}
                {get_text("usage_step1")}
                {get_text("usage_step2")}
                {get_text("usage_step3")}
                """,  
                f"✅ Language switched to {language.upper()} successfully! / 语言已成功切换为{language.upper()}！"  # 状态消息
            ]

        language_dropdown.change(
            fn=update_interface_language,
            inputs=[language_dropdown],
            outputs=[language_state, title_md, desc_md, txt_prompt, edit_instruction, understand_question, usage_md, txt_status]
        )

    return demo

demo = create_interface()


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8418,
        share=False,
        show_error=True
    )
