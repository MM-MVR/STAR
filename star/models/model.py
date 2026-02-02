import os
import math
import torch
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor

from star.models.config import STARMultiModalConfig
from star.models.pixel_encoder.vq_model import VQ_Model
from star.models.adapter.projector import MlpProjector
from star.models.pixel_decoder.lumina2_decoder import Lumina2Decoder
from star.models.data_process_utils import get_full_transform, get_vq_transform, preprocess_image_gen
from star.models.rope_2d import get_rope_index_25

class STARMultiModal(PreTrainedModel):
    def __init__(self, config: STARMultiModalConfig, args=None, **kwargs):
        super().__init__(config)

        self.config = config
        self.args = args if args is not None else kwargs.get("args", None)

        # Pixel Encoder Generation
        model_name = config.pixel_encoder.model_name
        if model_name == "VQ_Model":
            self.pixel_encoder = VQ_Model(config.pixel_encoder)
        else:
            assert None, f"Unsupported {model_name}"
        self.pixel_encoder.eval()


        # Pixel Adapter Generation
        model_name = config.pixel_adapter.model_name
        if model_name == "MLP_GELU":
            self.pixel_adapter = MlpProjector(config.pixel_adapter)
        else:
            assert None, f"Unsupported {model_name}"

        # Pixel Ouput Head Generation
        self.pixel_output_head = torch.nn.Linear(config.pixel_output_head.n_embed, config.pixel_output_head.image_token_size)

        if getattr(args, "diffusion_as_decoder") and args.diffusion_as_decoder:
            self.diffusion_decoder = Lumina2Decoder(config.pixel_decoder, args)
        else:
            self.diffusion_decoder = None
            
        # Large Language Model
        model_name, model_path = config.language_model.model_name, config.language_model.model_path
        
        if model_name == "Qwen2.5-VL":
            self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer

            self.image_processor = self.processor.image_processor 
            self.image_processor.max_pixels = self.args.max_pixels
            self.image_processor.min_pixels = self.args.min_pixels
            self.image_processor.size["longest_edge"] = self.args.max_pixels
            self.image_processor.size["shortest_edge"] = self.args.min_pixels

            special_token_tags = ["<|vision_start|>", "<|vision_pad|>", "<|image_pad|>", "<|vision_end|>", "<|fim_pad|>"]
            self.special_tokens = {tag: self.tokenizer.vocab.get(tag, None) for tag in special_token_tags}
            
        else:
            assert None, f"unsupported {model_name}: {model_path}"
        self.llm.generation_config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]

        if self.args.grad_ckpt:
            self.llm.gradient_checkpointing_enable()
            self.llm.visual.gradient_checkpointing_enable()


        stacked_ar_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers_to_extract = config.stacked_ar.num_layers
        stacked_ar_config.num_hidden_layers = num_layers_to_extract  
        
        self.stacked_ar = Qwen2_5_VLForConditionalGeneration(stacked_ar_config)
    
        temp_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        total_layers = len(temp_model.model.layers)
        start_layer = max(0, total_layers - num_layers_to_extract)
        temp_model.model.layers = temp_model.model.layers[start_layer:]
        self.stacked_ar.load_state_dict(temp_model.state_dict(), strict=False)

        self.stacked_ar = self.stacked_ar.to("cuda")
        del self.stacked_ar.visual, self.stacked_ar.model.embed_tokens, self.stacked_ar.lm_head


    # For Inference Generation
    def generate_images(self, prompt, max_new_tokens=256, num_return_sequences=1, cfg_weight=5.0, topk_sample=1000, topp_sample=1.0, temperature=1.0, reasoning=False, return_dict=False):
        
        if reasoning:
            return self.generate_images_reasoning(prompt, max_new_tokens, num_return_sequences, cfg_weight, topk_sample, topp_sample, temperature, return_dict)
        
        messages = [{'role': 'user', 'content': prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_token = self.tokenizer.encode(text)
        text_token = torch.tensor(text_token).long().to(self.device)
        
        keys = list(self.special_tokens.keys())
        start_token = (torch.ones(1) * self.special_tokens.get(keys[0])).long().to(self.device)
        
        input_ids = torch.cat((text_token, start_token)).long().to(self.device)
        tokens = torch.zeros((num_return_sequences*2, len(input_ids)), dtype=torch.int).cuda()
        assistant_tokens = input_ids[-4:]
        
        for i in range(num_return_sequences*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.special_tokens.get(keys[4])
                tokens[i, -4:] = assistant_tokens

        inputs_embeds = self.llm.model.embed_tokens(tokens).to(self.device)
        generated_tokens = torch.zeros((num_return_sequences, max_new_tokens), dtype=torch.int).cuda()

        for i in range(max_new_tokens):
            outputs = self.llm.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if i != 0 else None,
                    output_hidden_states=True)
            last_hidden_states = outputs[0]

            output_states = self.stacked_ar.model(
                inputs_embeds=last_hidden_states,
                past_key_values=output_states.past_key_values if i != 0 else None,
                output_hidden_states=True,
                use_cache=True)

            last_hidden_states = output_states.hidden_states[-1]

            logits = self.pixel_output_head(last_hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            next_token, _ = self.sample(logits, temperature=1.0, top_k=topk_sample, top_p=topp_sample)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            
            vqgan_embeds = self.pixel_encoder.get_codebook_entry(next_token)
            img_embeds = self.pixel_adapter(vqgan_embeds)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        latent_size = int(math.sqrt(max_new_tokens))
        output_images = self.pixel_encoder.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_return_sequences, self.pixel_encoder.config.codebook_embed_dim, latent_size, latent_size])
        output_images = output_images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        diff_images = None
        if self.diffusion_decoder is not None:
            gen_image_embeds = self.pixel_encoder.get_codebook_entry(generated_tokens)

            if self.args.diffusion_resolution==512:
                self.diffusion_decoder.pipe.transformer.config.sample_size=16
            elif self.args.diffusion_resolution==1024:
                self.diffusion_decoder.pipe.transformer.config.sample_size=32
            diff_images = self.diffusion_decoder.pipe(
                    prompt,
                    num_inference_steps=40,
                    guidance_scale=4.5,
                    gen_image_embeds=gen_image_embeds, #gen_image_embeds,
                    control_emd="text",
                    ori_inp_way=self.diffusion_decoder.transformer.ori_inp_dit,
                    only_t2i="vqconcat",
                    img_guidance_scale=1.05,
                    height=self.args.diffusion_resolution,
                    width=self.args.diffusion_resolution
                ).images
        if return_dict:
            return {"output_images": output_images, "generated_tokens": generated_tokens, "diff_images": diff_images}
        return output_images

    def answer_text_qwen_vl(self, question, max_new_tokens=256, do_sample=True):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.llm.device)
        
        # Inference: Generation of the output
        generated_ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else ""
    
    def generate_images_reasoning(self, prompt, max_new_tokens=256, num_return_sequences=1, cfg_weight=5.0, topk_sample=1000, topp_sample=1.0, temperature=1.0, return_dict=False):
        
        messages = [{'role': 'user', 'content': prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_token = self.tokenizer.encode(text)
        text_token = torch.tensor(text_token).long().to(self.device)
        
        keys = list(self.special_tokens.keys())
        start_token = (torch.ones(1) * self.special_tokens.get(keys[0])).long().to(self.device)
        
        input_ids = torch.cat((text_token, start_token)).long().to(self.device)
        tokens = torch.zeros((num_return_sequences*2, len(input_ids)), dtype=torch.int).cuda()
        assistant_tokens = input_ids[-4:]
        
        for i in range(num_return_sequences*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.special_tokens.get(keys[4])
                tokens[i, -4:] = assistant_tokens

        generated_tokens = torch.zeros((num_return_sequences, max_new_tokens), dtype=torch.int).cuda()
        answer_tokens_list = self.answer_text_qwen_vl(prompt, do_sample=False)
        
        if answer_tokens_list:
            answer_tokens_list = self.tokenizer.encode(answer_tokens_list, add_special_tokens=False)
            answer_tokens = torch.tensor([answer_tokens_list], device=self.device)  # [1, seq_len]
            magic_prompt = " Ultra HD, 4K, cinematic composition"

            
            magic_prompt_tokens = self.tokenizer.encode(magic_prompt, add_special_tokens=False)
            magic_prompt_tensor = torch.tensor([magic_prompt_tokens], device=self.device)  # [1, magic_seq_len]
            
            answer_tokens = torch.cat([answer_tokens, magic_prompt_tensor], dim=1)  # [1, seq_len + magic_seq_len]
            answer_prompt = self.tokenizer.decode(answer_tokens[0]).split("assistant\n")[-1] #hjc see
            
            special_token = self.special_tokens.get(keys[4])
            special_token_tensor = torch.tensor([[special_token]], device=self.device)
            special_token_expanded = special_token_tensor.expand(-1, answer_tokens.size(1))
            
            answer_tokens_with_special = torch.cat([answer_tokens, special_token_expanded], dim=0)
            
            batch_size = tokens.size(0)  # num_return_sequences*2
            answer_tokens_expanded = answer_tokens_with_special.repeat(batch_size // 2, 1)
            
            input_tokens = torch.cat((tokens[:, :14], answer_tokens_expanded, tokens[:, -6:]), dim=1)
            
        else:
            input_tokens = tokens
            answer_prompt = None
        
        inputs_embeds = self.llm.model.embed_tokens(input_tokens).to(self.device)

        for i in range(max_new_tokens):
            outputs = self.llm.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if i != 0 else None,
                    output_hidden_states=True)
            last_hidden_states = outputs[0]

            output_states = self.stacked_ar.model(
                inputs_embeds=last_hidden_states,
                past_key_values=output_states.past_key_values if i != 0 else None,
                output_hidden_states=True,
                use_cache=True)

            last_hidden_states = output_states.hidden_states[-1]

            logits = self.pixel_output_head(last_hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            next_token, _ = self.sample(logits, temperature=1.0, top_k=topk_sample, top_p=topp_sample)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            
            vqgan_embeds = self.pixel_encoder.get_codebook_entry(next_token)
            img_embeds = self.pixel_adapter(vqgan_embeds)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        latent_size = int(math.sqrt(max_new_tokens))
        output_images = self.pixel_encoder.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_return_sequences, self.pixel_encoder.config.codebook_embed_dim, latent_size, latent_size])
        output_images = output_images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        diff_images = None
        if self.diffusion_decoder is not None:
            gen_image_embeds = self.pixel_encoder.get_codebook_entry(generated_tokens)
            diff_prompt = answer_prompt if answer_prompt else prompt
            if self.args.diffusion_resolution==512:
                self.diffusion_decoder.pipe.transformer.config.sample_size=16
            elif self.args.diffusion_resolution==1024:
                self.diffusion_decoder.pipe.transformer.config.sample_size=32
            diff_images = self.diffusion_decoder.pipe(
                    diff_prompt,
                    num_inference_steps=40,
                    guidance_scale=4.5,
                    gen_image_embeds=gen_image_embeds, #gen_image_embeds,
                    control_emd="text",
                    ori_inp_way=self.diffusion_decoder.transformer.ori_inp_dit,
                    only_t2i="vqconcat",
                    img_guidance_scale=1.05,
                    height=self.args.diffusion_resolution,
                    width=self.args.diffusion_resolution
                ).images
        if return_dict:
            return {"output_images":output_images,"generated_tokens":generated_tokens,"diff_images":diff_images,"answer_prompt":answer_prompt}
        return output_images
    
    def generate_images_edit(self, image, prompt, max_new_tokens=256, num_return_sequences=1, cfg_weight=5.0, topk_sample=1000, topp_sample=1.0, temperature=1.0,return_dict=False):

        vq_image_transform = get_vq_transform(self.args)
        full_image_transform = get_full_transform(self.args)

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, list):
            image = [each_image.convert('RGB') for each_image in image]
        else:
            image = image.convert('RGB')

        messages = [{'role': 'user', 'content': prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_token = self.tokenizer.encode(text)
        text_token = torch.tensor(text_token).long().to(self.device)
        
        keys = list(self.special_tokens.keys())
        start_token = (torch.ones(1) * self.special_tokens.get(keys[0])).long().to(self.device)
        user_prompt = "<|im_start|>user\n"
        user_prompt_token = self.tokenizer.encode(user_prompt, add_special_tokens=False)
        user_prompt_tensor = torch.tensor(user_prompt_token).long().to(self.device)
        windows = text_token.unfold(0, len(user_prompt_tensor), 1)
        matches = (windows == user_prompt_tensor).all(dim=1)
        image_position = torch.where(matches)[0][0].item() + len(user_prompt_tensor)

        input_ids = torch.cat((text_token, start_token)).long().to(self.device)
        tokens = torch.zeros((num_return_sequences*2, len(input_ids)), dtype=torch.int).cuda()
        assistant_tokens = input_ids[-4:]
        
        for i in range(num_return_sequences*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.special_tokens.get(keys[4])
                tokens[i, -4:] = assistant_tokens

        inputs_embeds = self.llm.model.embed_tokens(tokens).to(self.device)
        position_ids = None

        if image is not None:
            image_info = preprocess_image_gen(image, self.image_processor, vq_image_transform)
            image_embeds = self.llm.visual(image_info["pixel_values"].to(inputs_embeds.device,self.llm.visual.dtype), grid_thw=image_info["image_grid_thw"].to(inputs_embeds.device))
            image_embeds = image_embeds[None,:].repeat(2, 1, 1).to(inputs_embeds.device, inputs_embeds.dtype)
            
            vq_pixel_values = image_info["vq_pixel_values"].to(inputs_embeds.device)
            B = inputs_embeds.size(0)
            if len(vq_pixel_values.shape)==4:
                vq_pixel_values = vq_pixel_values[:,None]
            N = vq_pixel_values.size(1)
            _, _, [_, _, vq_indices] = self.pixel_encoder.encode(vq_pixel_values.flatten(0, 1).bfloat16())
            batch_size = vq_pixel_values.shape[0]
            vq_indices = vq_indices.reshape(batch_size, N, vq_indices.shape[-1])
            vqgan_dec_embeds = self.pixel_encoder.get_codebook_entry(vq_indices)
            vq_embeds = self.pixel_adapter(vqgan_dec_embeds)
            vq_embeds = vq_embeds.repeat(B, 1, 1, 1).to(inputs_embeds.device, inputs_embeds.dtype).flatten(1, 2)

            vision_start_embeds = self.llm.model.embed_tokens(torch.tensor(self.tokenizer.encode("<|vision_start|>")).long().to(self.device))
            vision_end_embeds = self.llm.model.embed_tokens(torch.tensor(self.tokenizer.encode("<|vision_end|>")).long().to(self.device))
            newline_embeds = self.llm.model.embed_tokens(torch.tensor(self.tokenizer.encode("\n")).long().to(self.device))
            vision_start_embeds = vision_start_embeds.unsqueeze(0).repeat(B, 1, 1)
            vision_end_embeds = vision_end_embeds.unsqueeze(0).repeat(B, 1, 1)
            newline_embeds = newline_embeds.unsqueeze(0).repeat(B, 1, 1)

            inputs_embeds = torch.cat((inputs_embeds[:, :image_position], 
                                       vision_start_embeds, vq_embeds, vision_end_embeds, 
                                       vision_start_embeds, image_embeds, vision_end_embeds, newline_embeds, 
                                       inputs_embeds[:, image_position:]), dim=1)
            
            SPECIAL_VQ_TOKEN = '<|vision_pad|>'    
            SPECIAL_VIT_TOKEN = '<|image_pad|>'
            SPECIAL_VQ_TOKEN_ID = self.tokenizer.encode(SPECIAL_VQ_TOKEN)[0]
            SPECIAL_VIT_TOKEN_ID = self.tokenizer.encode(SPECIAL_VIT_TOKEN)[0]
            input_ids_for_position = torch.cat([input_ids[:image_position], 
                                       torch.tensor(self.tokenizer.encode("<|vision_start|>")).to(vq_embeds.device), torch.full((vq_embeds.shape[-2],), SPECIAL_VQ_TOKEN_ID, device=vq_embeds.device), torch.tensor(self.tokenizer.encode("<|vision_end|>")).to(vq_embeds.device), 
                                       torch.tensor(self.tokenizer.encode("<|vision_start|>")).to(vq_embeds.device),  torch.full((image_embeds.shape[-2],), SPECIAL_VIT_TOKEN_ID, device=vq_embeds.device), torch.tensor(self.tokenizer.encode("<|vision_end|>")).to(vq_embeds.device), torch.tensor(self.tokenizer.encode("\n")).to(vq_embeds.device), 
                                       input_ids[image_position:],torch.full((vq_embeds.shape[-2],), SPECIAL_VQ_TOKEN_ID, device=vq_embeds.device)], dim=0)
            position_ids, _ = get_rope_index_25(
                self.image_processor.merge_size,
                input_ids_for_position[None],
                image_grid_thw=image_info["image_grid_thw"],
                video_grid_thw=None,
                second_per_grid_ts=None,
            )
            
        generated_tokens = torch.zeros((num_return_sequences, max_new_tokens), dtype=torch.int).cuda()
        
        for i in range(max_new_tokens):
            if i != 0:
                real_position = position_ids[:,:,outputs.past_key_values.seen_tokens:(outputs.past_key_values.seen_tokens+inputs_embeds.shape[1])].to(inputs_embeds.device)
            else:
                real_position = position_ids[:,:,:inputs_embeds.shape[1]].to(inputs_embeds.device)
            outputs = self.llm.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True, 
                    position_ids = real_position,
                    past_key_values=outputs.past_key_values if i != 0 else None,
                    output_hidden_states=True)
            last_hidden_states = outputs[0]

            output_states = self.stacked_ar.model(
                inputs_embeds=last_hidden_states,
                past_key_values=output_states.past_key_values if i != 0 else None,
                output_hidden_states=True,
                position_ids = real_position,
                use_cache=True)

            last_hidden_states = output_states.hidden_states[-1]

            logits = self.pixel_output_head(last_hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            next_token, _ = self.sample(logits, temperature=1.0, top_k=topk_sample, top_p=topp_sample)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            

            vqgan_embeds = self.pixel_encoder.get_codebook_entry(next_token)
            img_embeds = self.pixel_adapter(vqgan_embeds)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        latent_size = int(math.sqrt(max_new_tokens))
        output_images = self.pixel_encoder.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_return_sequences, self.pixel_encoder.config.codebook_embed_dim, latent_size, latent_size])
        output_images = output_images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            
        diff_images = None
        if self.diffusion_decoder is not None:

            gen_image_embeds = self.pixel_encoder.get_codebook_entry(generated_tokens)
            
            if isinstance(image, list):
                processed_img = [full_image_transform(each_image) for each_image in image]
            else:
                processed_img = [full_image_transform(image)]
            if self.args.diffusion_resolution==512:
                self.diffusion_decoder.pipe.transformer.config.sample_size=16
            elif self.args.diffusion_resolution==1024:
                self.diffusion_decoder.pipe.transformer.config.sample_size=32
            diff_images = self.diffusion_decoder.pipe(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=3.0,
                    gen_image_embeds=gen_image_embeds, #gen_image_embeds,
                    control_emd="text",ori_inp_img=processed_img[0],ori_inp_way="seq",
                    only_t2i="vqconcat",img_guidance_scale=1.8,vq_guidance_scale=1,height=self.args.diffusion_resolution,width=self.args.diffusion_resolution
                ).images
        if return_dict:
            return {"output_images": output_images, "generated_tokens": None, "diff_images": diff_images}
        return None
    
    def sample(self, logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
        
        logits = logits / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_logits:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
        return idx, probs

    def top_k_top_p_filtering(
        self,
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    # For Inference Understand
    def preprocess_image(self, image):
        if image is None:
            return None
        if isinstance(image, str):
            if os.path.exists(image):
                pil_image = Image.open(image).convert('RGB')
            else:
                response = requests.get(image)
                if response.status_code == 200:
                    image_bytes = BytesIO(response.content)
                    pil_image = Image.open(image_bytes).convert('RGB')
                else:
                    raise ValueError(f"Failed to load image from url {image}")
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        elif isinstance(image, list):
            return self.preprocess_image(image[0])
        else:
            raise ValueError("Unsupported image type")
        
        return pil_image

    def inference_understand(self, image, question, max_new_tokens=256):
        pil_image = self.preprocess_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        from qwen_vl_utils import process_vision_info
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.llm.device)

        # Inference: Generation of the output
        generated_ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else ""
