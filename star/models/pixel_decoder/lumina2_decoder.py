import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, Lumina2Pipeline 
from transformers import AutoTokenizer, Gemma2Model
import copy
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.pipelines.lumina2.pipeline_lumina2 import *

class Lumina2Decoder(torch.nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.diffusion_model_path = config.model_path

        if not hasattr(args, "revision"):    
            args.revision = None
        if not hasattr(args, "variant"):    
            args.variant = None
            
        self.tokenizer_one = AutoTokenizer.from_pretrained(
                self.diffusion_model_path,
                subfolder="tokenizer",
                revision=args.revision,
            )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.diffusion_model_path, subfolder="scheduler"
                )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.text_encoder_one = Gemma2Model.from_pretrained(
                self.diffusion_model_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
            )
        self.text_encoding_pipeline = Lumina2Pipeline.from_pretrained(
            self.diffusion_model_path,
            vae=None,
            transformer=None,
            text_encoder=self.text_encoder_one,
            tokenizer=self.tokenizer_one,
            )
        self.vae = AutoencoderKL.from_pretrained(
                    self.diffusion_model_path,
                    subfolder="vae",
                    revision=args.revision,
                    variant=args.variant,
                )
        if args.ori_inp_dit=="seq":
            from star.models.pixel_decoder.transformer_lumina2_seq import Lumina2Transformer2DModel
        elif args.ori_inp_dit=="ref":
            from star.models.pixel_decoder.transformer_lumina2 import Lumina2Transformer2DModel

        self.transformer = Lumina2Transformer2DModel.from_pretrained(
                self.diffusion_model_path, subfolder="transformer", revision=args.revision, variant=args.variant
            )
        
        vq_dim = 512
        patch_size = self.transformer.config.patch_size
        in_channels = vq_dim + self.transformer.config.in_channels # 48 for mask
        out_channels = self.transformer.x_embedder.out_features

        load_num_channel = self.transformer.config.in_channels * patch_size * patch_size
        self.transformer.register_to_config(in_channels=in_channels)
        transformer = self.transformer
        with torch.no_grad():
            new_proj = nn.Linear(
                in_channels * patch_size * patch_size, out_channels, bias=True
            )

            new_proj.weight.zero_()

            new_proj = new_proj.to(transformer.x_embedder.weight.dtype)
            new_proj.weight[:, :load_num_channel].copy_(transformer.x_embedder.weight)
            new_proj.bias.copy_(transformer.x_embedder.bias)
            transformer.x_embedder = new_proj
            
        self.ori_inp_dit = args.ori_inp_dit
        if args.ori_inp_dit=="seq":
            refiner_channels = transformer.noise_refiner[-1].dim 
            with torch.no_grad():
                vae2cond_proj1 = nn.Linear(refiner_channels, refiner_channels, bias=True)
                vae2cond_act = nn.GELU(approximate='tanh')
                vae2cond_proj2 = nn.Linear(refiner_channels, refiner_channels, bias=False)
                vae2cond_proj2.weight.zero_()
                
                ori_inp_refiner = nn.Sequential(
                    vae2cond_proj1,
                    vae2cond_act,
                    vae2cond_proj2
                )
                transformer.ori_inp_refiner = ori_inp_refiner
                transformer.ori_inp_dit = self.ori_inp_dit
        elif args.ori_inp_dit=="ref":
            transformer.initialize_ref_weights()
            transformer.ori_inp_dit = self.ori_inp_dit

        transformer.requires_grad_(True)
        
        if args.grad_ckpt and args.diffusion_resolution==1024:
            transformer.gradient_checkpointing = args.grad_ckpt
            transformer.enable_gradient_checkpointing()
            
        self.vae.requires_grad_(False)
        self.vae.to(dtype=torch.float32)
        self.args = args
        
        self.pipe = Lumina2InstructPix2PixPipeline.from_pretrained(self.diffusion_model_path, 
                                                                transformer=transformer,
                                                                text_encoder=self.text_encoder_one,
                                                                vae=self.vae,
                                                                torch_dtype=torch.bfloat16)

    
        with torch.no_grad():
            _, _, self.uncond_prompt_embeds, self.uncond_prompt_attention_mask = self.text_encoding_pipeline.encode_prompt(
                "",
                max_sequence_length=self.args.max_diff_seq_length,
            )

    def compute_text_embeddings(self,prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                prompt,
                max_sequence_length=self.args.max_diff_seq_length,
            )
        return prompt_embeds, prompt_attention_mask
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device=timesteps.device)
        timesteps = timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def forward(self, batch_gpu,batch, image_embeds):
        args = self.args
        pixel_values = batch_gpu["full_pixel_values"].to(dtype=self.vae.dtype) #aux_image
        data_type = "t2i"
        if len(pixel_values.shape)==5:
            bs,num_img,c,h,w = pixel_values.shape
            if num_img==2:
                data_type = "edit"
            pixel_values_ori_img = pixel_values[:,0]
            pixel_values = pixel_values[:,-1]
        pixel_values = F.interpolate(pixel_values, size=(self.args.diffusion_resolution, self.args.diffusion_resolution), mode='bilinear',align_corners=False)
        if data_type=="edit" and self.ori_inp_dit!="none":
            pixel_values_ori_img = F.interpolate(pixel_values_ori_img, size=(self.args.diffusion_resolution, self.args.diffusion_resolution), mode='bilinear', align_corners=False)
        prompt = batch["prompts"]
        bs,_,_,_ = pixel_values.shape
        image_prompt_embeds = None
        image_embeds_2d = image_embeds.reshape(bs, 24, 24, image_embeds.shape[-1]).permute(0, 3, 1, 2)
        image_embeds_2d = F.interpolate(image_embeds_2d, size=(args.diffusion_resolution//8, args.diffusion_resolution//8), mode='bilinear', align_corners=False)
        
        control_emd = args.control_emd
        prompt_embeds, prompt_attention_mask = self.compute_text_embeddings(prompt, self.text_encoding_pipeline)
        if control_emd=="mix":
            prompt_embeds=torch.cat([prompt_embeds, image_prompt_embeds], dim=1) #use mix
        elif control_emd=="null":
            prompt_embeds = torch.zeros_like(prompt_embeds)
            prompt_attention_mask = torch.ones_like(prompt_attention_mask)
        elif control_emd=="text":
            pass
        elif control_emd=="vit" or control_emd=="vq" or control_emd=="vqvae" or control_emd=="vqconcat" or control_emd=="vqconcatvit":
            prompt_embeds=image_prompt_embeds
        
        
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = latents.to(dtype=image_embeds.dtype)

        latents_ori_img = torch.zeros_like(latents)
        if data_type=="edit" and self.ori_inp_dit!="none":
            latents_ori_img = self.vae.encode(pixel_values_ori_img).latent_dist.sample()
            latents_ori_img = (latents_ori_img - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents_ori_img = latents_ori_img.to(dtype=image_embeds.dtype)
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )

        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype).to(device=noise.device)
        #noisy_model_input = (1.0 - sigmas) * noise + sigmas * latents
        noisy_model_input = sigmas * noise + (1-sigmas) * latents
        #noisy_model_input + (1-sigmas)*(latents - noise) = latents
        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_image_embeds = image_embeds_2d
                
        if args.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * args.uncondition_prob 
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            #prompt_embeds = torch.where(prompt_mask, torch.zeros_like(prompt_embeds), prompt_embeds)
            prompt_embeds = torch.where(prompt_mask, self.uncond_prompt_embeds.repeat(prompt_embeds.shape[0],1,1).to(prompt_embeds.device), prompt_embeds)
            prompt_attention_mask = torch.where(prompt_mask[:,0], self.uncond_prompt_attention_mask.repeat(prompt_embeds.shape[0],1).to(prompt_embeds.device), prompt_attention_mask)
                        
            # Sample masks for the original images.
            #random_p_vq = torch.rand(bsz, device=latents.device)
            image_mask_dtype = original_image_embeds.dtype
            image_mask = 1 - (
                (random_p <= args.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            
            if data_type=="edit":
                image_mask=0 
            # Final image conditioning.
            original_image_embeds = image_mask * original_image_embeds
            
            ori_latent_mask = 1 - (
                (random_p >= args.uncondition_prob).to(image_mask_dtype)
                * (random_p < 3 * args.uncondition_prob).to(image_mask_dtype)
            )
            ori_latent_mask = ori_latent_mask.reshape(bsz, 1, 1, 1)
            latents_ori_img = ori_latent_mask * latents_ori_img
            
        concatenated_noisy_latents = torch.cat([noisy_model_input, original_image_embeds], dim=1)

        ref_image_hidden_states = None
        if self.ori_inp_dit=="dim":
            concatenated_noisy_latents = torch.cat([concatenated_noisy_latents, latents_ori_img], dim=1)
        elif self.ori_inp_dit=="seq":
            latents_ori_img = torch.cat([latents_ori_img, original_image_embeds], dim=1)
            concatenated_noisy_latents = torch.cat([concatenated_noisy_latents, latents_ori_img], dim=2)
        elif self.ori_inp_dit=="ref":
            latents_ori_img = torch.cat([latents_ori_img, original_image_embeds], dim=1)
            ref_image_hidden_states = latents_ori_img[:,None]
        # Predict the noise residual
        # scale the timesteps (reversal not needed as we used a reverse lerp above already)
        timesteps = 1-timesteps / self.noise_scheduler.config.num_train_timesteps #timesteps / self.noise_scheduler.config.num_train_timesteps
        model_pred = self.transformer(
            hidden_states=concatenated_noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            # ref_image_hidden_states = ref_image_hidden_states,
            return_dict=False,
        )[0]
        if self.ori_inp_dit=="seq":
            model_pred = model_pred[:, :, :args.diffusion_resolution//8, :]
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        target = latents - noise
        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.

        # Concatenate the `original_image_embeds` with the `noisy_latents`.

        # Get the target for loss depending on the prediction type
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        
        loss_value = loss.item()
                
        return loss


class Lumina2InstructPix2PixPipeline(Lumina2Pipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        negative_prompt: Union[str, List[str]] = None,
        sigmas: List[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        system_prompt: Optional[str] = None,
        cfg_trunc_ratio=[0.0,1.0],
        cfg_normalization: bool = False,
        max_sequence_length: int = 256,
        control_emd="text",
        img_cfg_trunc_ratio =[0.0,1.0],
        gen_image_embeds=None,only_t2i="vqconcat",image_prompt_embeds=None,ori_inp_img=None,img_guidance_scale=1.5,vq_guidance_scale=0,ori_inp_way="none",
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        num_images_per_prompt = gen_image_embeds.shape[0] if gen_image_embeds is not None else image_prompt_embeds.shape[0]
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            system_prompt=system_prompt,
        )
        

        if gen_image_embeds is not None: 
            image_embeds_8=gen_image_embeds
        
        if control_emd=="text":
            pass
        elif control_emd=="null":
            prompt_embeds = torch.zeros_like(prompt_embeds)
            prompt_attention_mask = torch.zeros_like(prompt_attention_mask)
            negative_prompt_embeds = prompt_embeds
            negative_prompt_attention_mask = prompt_attention_mask
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds,negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask,negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        # 4. Prepare latents.
        latent_channels = self.vae.config.latent_channels #self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        latents_ori_img = torch.zeros_like(latents)
        if ori_inp_img is not None and ori_inp_way !="none":
            #fuck =  torch.load(ori_inp_img).to(latents.device)
            ori_inp_img = F.interpolate(ori_inp_img[None].to(latents.device,latents.dtype), size=(height,width), mode='bilinear',align_corners=False)
            latents_ori_img = self.vae.encode(ori_inp_img).latent_dist.sample()
            latents_ori_img = (latents_ori_img- self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents_ori_img = latents_ori_img.to(dtype=latents.dtype)
        if ori_inp_way !="none":    
            negative_latents_ori_img = torch.zeros_like(latents_ori_img).to(prompt_embeds.dtype)
            latents_ori_img = torch.cat([negative_latents_ori_img,latents_ori_img, latents_ori_img], dim=0) if self.do_classifier_free_guidance else latents_ori_img
        
        vq_in_edit = False
        if only_t2i==True:
            image_latents = torch.zeros_like(latents)[:,:8]
        elif only_t2i=="vqconcat":
            image_embeds_2d = image_embeds_8.reshape(batch_size* num_images_per_prompt,24,24,image_embeds_8.shape[-1]).permute(0,3,1,2)
            if ori_inp_img is not None and image_embeds_8.mean()!=0:
                vq_in_edit = True
                image_vq_latents = F.interpolate(image_embeds_2d, size=(height//8,width//8), mode='bilinear',align_corners=False).to(latents.device,latents.dtype)
                image_latents = torch.zeros_like(image_vq_latents)
            else:
                image_latents = F.interpolate(image_embeds_2d, size=(height//8,width//8), mode='bilinear',align_corners=False).to(latents.device,latents.dtype)
        
        negative_image_latents = torch.zeros_like(image_latents).to(prompt_embeds.dtype)
        image_latents = torch.cat([negative_image_latents,image_latents, image_latents], dim=0) if self.do_classifier_free_guidance else image_latents
        
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        self.scheduler.sigmas=self.scheduler.sigmas.to(latents.dtype) #hjc find bug
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # compute whether apply classifier-free truncation on this timestep
                do_classifier_free_truncation = not ((i + 1) / num_inference_steps > cfg_trunc_ratio[0] and (i + 1) / num_inference_steps < cfg_trunc_ratio[1])
                img_do_classifier_free_truncation = not ((i + 1) / num_inference_steps > img_cfg_trunc_ratio[0] and (i + 1) / num_inference_steps < img_cfg_trunc_ratio[1])
                
                # reverse the timestep since Lumina uses t=0 as the noise and t=1 as the image
                current_timestep = 1 - t / self.scheduler.config.num_train_timesteps
                
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                
                ref_image_hidden_states = None
                if ori_inp_way=="seq":
                    latents_ori_img_cat = torch.cat([latents_ori_img, image_latents], dim=1)
                    latent_model_input = torch.cat([latent_model_input, latents_ori_img_cat], dim=2)
                elif ori_inp_way=="ref":
                    latents_ori_img_cat = torch.cat([latents_ori_img, image_latents], dim=1)
                    ref_image_hidden_states = latents_ori_img_cat[:,None]
                
                if ori_inp_way=="ref":
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=current_timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        return_dict=False,ref_image_hidden_states=ref_image_hidden_states,
                        attention_kwargs=self.attention_kwargs,
                    )[0]
                else:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=current_timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        return_dict=False,
                        attention_kwargs=self.attention_kwargs,
                    )[0]
                if ori_inp_way=="seq":
                    noise_pred = noise_pred[:,:,:height//8,:]

                if vq_in_edit:
                    latent_model_vq_input = torch.cat([latents, image_vq_latents], dim=1)
                    if ori_inp_way=="seq":
                        latents_ori_img_cat_vq = torch.cat([torch.zeros_like(latents), image_vq_latents], dim=1)
                        latent_model_vq_input = torch.cat([latent_model_vq_input, latents_ori_img_cat_vq], dim=2)
                    
                    noise_vq_pred = self.transformer(
                        hidden_states=latent_model_vq_input,
                        timestep=current_timestep[-1:],
                        encoder_hidden_states=prompt_embeds[-1:],
                        encoder_attention_mask=prompt_attention_mask[-1:],
                        return_dict=False,
                        attention_kwargs=self.attention_kwargs,
                    )[0]
                    if ori_inp_way=="seq":
                        noise_vq_pred = noise_vq_pred[:,:,:height//8,:]
                # perform normalization-based guidance scale on a truncated timestep interval
                if self.do_classifier_free_guidance:
                    noise_pred_uncond,noise_pred_img, noise_pred_text = noise_pred.chunk(3)
                    if not do_classifier_free_truncation and not img_do_classifier_free_truncation:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_img)+ img_guidance_scale * (noise_pred_img - noise_pred_uncond)
                    elif not do_classifier_free_truncation and img_do_classifier_free_truncation:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_img)+ 1 * (noise_pred_img - noise_pred_uncond)
                    elif do_classifier_free_truncation and not img_do_classifier_free_truncation:
                        noise_pred = noise_pred_uncond + 1 * (noise_pred_text - noise_pred_img)+ img_guidance_scale * (noise_pred_img - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_text
                    if vq_in_edit:
                        noise_pred = noise_pred +vq_guidance_scale*(noise_vq_pred-noise_pred_uncond)
                    # apply normalization after classifier-free guidance
                    if cfg_normalization:
                        cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_pred = noise_pred * (cond_norm / noise_norm)
                else:
                    noise_pred = noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                noise_pred = -noise_pred
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    