# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.fm_solvers_origin import FlowMatchScheduler
from .utils.fm_solvers_modified import FlowMatchNewScheduler


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 
                 # <<< MODIFICATION START >>>
                 # 新增参数: 保存中间变量latent
                 save_intermediate_latents=False,
                 save_intermediate_latent_path="intermediate_latents.pt",
                 load_intermediate_latent_path=None,      
                 load_intermediate_latent_t=None,
                 latent_output_dir=None,
                 # <<< MODIFICATION END >>>
                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            round(np.sqrt(max_area * aspect_ratio)) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        # <<< MODIFICATION START: 处理 initial_latent 或生成 noise >>>
        expected_latent_shape = (16, 21, lat_h, lat_w)
        if load_intermediate_latent_path is not None:
            num_t = load_intermediate_latent_t
            intermediate_data = torch.load(load_intermediate_latent_path, map_location='cpu')
            initial_latent = intermediate_data[num_t]
            assert initial_latent.shape == expected_latent_shape
            # 使用提供的 initial_latent 作为起始点
            latent = initial_latent.to(device=self.device, dtype=torch.float32)
            if self.rank == 0:
                print(f"Using provided initial latent with shape {latent.shape}.")
        else:
            latent = torch.randn(
                16,
                21,
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=self.device)
            if self.rank == 0:
                print(f"Generated initial random noise with shape {latent.shape} using seed {seed}.")
        # <<< MODIFICATION END >>>
        
        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        # <<< MODIFICATION START: 初始化用于存储中间latent的字典 >>>
        save_intermediate_latents_data = {}
        # <<< MODIFICATION END >>>

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            elif sample_solver == 'fm':
                sample_scheduler = FlowMatchScheduler(
                    num_inference_steps=sampling_steps, 
                    num_train_timesteps=self.num_train_timesteps,
                    shift=5.0,
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'fm_new':
                sample_scheduler = FlowMatchNewScheduler(
                    num_inference_steps=51,
                    num_train_timesteps=self.num_train_timesteps,
                    shift=5.0,
                )
                # if load_intermediate_latent_t is not None:
                #     timestep_id = torch.argmin((sample_scheduler.timesteps - load_intermediate_latent_t).abs())
                #     sample_scheduler.timesteps = sample_scheduler.timesteps[timestep_id:]
                timesteps = sample_scheduler.timesteps[:-1]
                if self.rank == 0:
                    print(sample_scheduler.timesteps)
                
            else:
                raise NotImplementedError("Unsupported solver.")
            
            if self.rank == 0:
                print("Scheduler: ", sample_solver)

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            
            for progress_id, t in enumerate(tqdm(timesteps)):
                # <<< MODIFICATION START: 保存当前的 latent (即x_t) >>>
                if save_intermediate_latents:
                    save_intermediate_latents_data[t.item()] = latent.clone().cpu()
                # <<< MODIFICATION END >>>
                
                # modified: batch input
                
                latent_model_input = [latent.to(self.device), latent.to(self.device)]
                timestep = torch.stack([t, t]).to(self.device) 
                
                context_list = [context[0], context_null[0]]
                clip_fea = torch.cat([clip_context, clip_context], dim=0)
                y_list = [y, y]
                
                noise_preds_list = self.model(
                    x=latent_model_input,
                    t=timestep,
                    context=context_list,
                    seq_len=max_seq_len, # seq_len remains the same for padding
                    clip_fea=clip_fea,
                    y=y_list,
                    progress_id=progress_id, # <--- 新增传递参数
                    sampling_steps=sampling_steps, # <--- 新增传递参数
                    latent_output_dir=latent_output_dir,
                ) 
                
                if offload_model:
                     torch.cuda.empty_cache()
                
                noise_pred_cond = noise_preds_list[0].to(torch.device('cpu') if offload_model else self.device)
                noise_pred_uncond = noise_preds_list[1].to(torch.device('cpu') if offload_model else self.device)
                
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)
                
                if sample_solver == "fm_new":
                    latents_mid = sample_scheduler.step_mid(noise_pred.unsqueeze(0), 
                                                            torch.stack([t]).to(self.device), latent.unsqueeze(0))
                    
                    #print(latents_mid.shape)
                    
                    latent_model_input_mid = [latents_mid[0].to(self.device), latents_mid[0].to(self.device)]
                    
                    t_mid = (torch.stack([t]) + sample_scheduler.timesteps[progress_id + 1]) / 2
                    timestep_mid = torch.stack([t_mid[0], t_mid[0]]).to(self.device) 
                    
                    noise_preds_list_mid = self.model(
                        x=latent_model_input_mid,
                        t=timestep_mid,
                        context=context_list,
                        seq_len=max_seq_len,
                        clip_fea=clip_fea,
                        y=y_list,
                        latent_output_dir=None,
                    ) 
                    
                    if offload_model:
                        torch.cuda.empty_cache()
                    
                    noise_pred_mid_posi = noise_preds_list_mid[0].to(torch.device('cpu') if offload_model else self.device)
                    noise_pred_mid_nega = noise_preds_list_mid[1].to(torch.device('cpu') if offload_model else self.device)
                    noise_pred_mid = noise_pred_mid_nega + guide_scale * (noise_pred_mid_posi - noise_pred_mid_nega)
                    
                    latent = sample_scheduler.step_solver(noise_pred_mid.unsqueeze(0), noise_pred.unsqueeze(0), 
                                                          torch.stack([t]).to(self.device), latent.unsqueeze(0)).squeeze(0)
                    del latent_model_input_mid
                    
                    if self.rank == 0:
                        temp_video = self.vae.decode([latent.to(self.device)])
                        import os
                        os.makedirs(f"/nvme2/vision/Wan2.1/intermediate_videos/{latent_output_dir}", exist_ok=True)
                        torch.save(temp_video[0], f"/nvme2/vision/Wan2.1/intermediate_videos/{latent_output_dir}/{progress_id}.pt")
                    
                    
                else:       
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        # <<< MODIFICATION START: 将保存的中间 latent 写入文件 >>>
        if save_intermediate_latents and self.rank == 0: # 只有 rank 0 进程执行保存操作
            try:
                print(f"Saving {len(save_intermediate_latents_data)} intermediate latents to {save_intermediate_latent_path}")
                torch.save(save_intermediate_latents_data, save_intermediate_latent_path)
                print("Intermediate latents saved successfully.")
            except Exception as e:
                print(f"Error saving intermediate latents: {e}")
        # <<< MODIFICATION END >>>
        
        del latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def generate_with_pnp(self,
                 input_prompt,
                 input_prompt_origin, 
                 img,
                 img_origin,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 pnp_layers=None,
                 load_intermediate_latent_path=None,      
                 load_intermediate_latent_t=None,
                 injection_step=None,
                 latent_output_dir=None,
                 is_delete=False,
                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        if self.rank == 0:
            print("PnP init!")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        img_origin = TF.to_tensor(img_origin).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            round(np.sqrt(max_area * aspect_ratio)) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        expected_latent_shape = (16, 21, lat_h, lat_w) if not is_delete else (16, 13, lat_h, lat_w)
        
        num_t = load_intermediate_latent_t
        intermediate_data = torch.load(load_intermediate_latent_path, map_location='cpu')
        initial_latent = intermediate_data[num_t]
        assert initial_latent.shape == expected_latent_shape
        # 使用提供的 initial_latent 作为起始点
        latent = initial_latent.to(device=self.device, dtype=torch.float32)
        
        if self.rank == 0:
            print(f"Using provided initial latent with shape {latent.shape}.")
                
        latent_origin = latent.clone().detach().to(device=self.device, dtype=torch.float32)
        
        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device) if not is_delete else torch.ones(1, 49, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
        dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_origin = self.text_encoder([input_prompt_origin], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_origin = self.text_encoder([input_prompt_origin], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_origin = [t.to(self.device) for t in context_origin] 
            context_null = [t.to(self.device) for t in context_null]
            
        self.clip.model.to(self.device)
        clip_context_origin = self.clip.visual([img_origin[:, None, :, :]])
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        # 修改前照片
        y1 = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img_origin[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                         dim=1).to(self.device) 
        ])[0] if not is_delete else self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img_origin[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 48, h, w)
            ],
                         dim=1).to(self.device) 
        ])[0]
        y1 = torch.concat([msk, y1])
        
        # 修改后照片
        y2 = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                         dim=1).to(self.device)
        ])[0] if not is_delete else self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 48, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y2 = torch.concat([msk, y2])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            elif sample_solver == 'fm':
                sample_scheduler = FlowMatchScheduler(
                    num_inference_steps=sampling_steps, 
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'fm_new':
                sample_scheduler = FlowMatchNewScheduler(
                    num_inference_steps=sampling_steps,
                    num_train_timesteps=self.num_train_timesteps,
                    shift=5.0,
                )
                timesteps = sample_scheduler.timesteps[:-1]  
            else:
                raise NotImplementedError("Unsupported solver.")
            
            if self.rank == 0:
                print("Scheduler: ", sample_solver)

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            
            for progress_id, t in enumerate(tqdm(timesteps)):
                
                latent_model_input = [latent_origin.to(self.device),
                                      latent.to(self.device), latent.to(self.device)]
                timestep = torch.stack([t, t, t]).to(self.device) 
                
                context_list = [context_origin[0], context[0], context_null[0]]
                clip_fea = torch.cat([clip_context_origin,clip_context, clip_context], dim=0)
                y_list = [y1, y2, y2]
                
                noise_preds_list = self.model(
                    x=latent_model_input,
                    t=timestep,
                    context=context_list,
                    seq_len=max_seq_len, # seq_len remains the same for padding
                    clip_fea=clip_fea,
                    y=y_list,
                    pnp=True,
                    pnp_layers=pnp_layers,
                    progress_id=progress_id, # <--- 新增传递参数
                    sampling_steps=sampling_steps, # <--- 新增传递参数
                    injection_step=injection_step,
                    latent_output_dir=latent_output_dir,
                ) 
                
                if offload_model:
                     torch.cuda.empty_cache()
                
                noise_pred_origin = noise_preds_list[0].to(torch.device('cpu') if offload_model else self.device)
                
                noise_pred_cond = noise_preds_list[1].to(torch.device('cpu') if offload_model else self.device)
                noise_pred_uncond = noise_preds_list[2].to(torch.device('cpu') if offload_model else self.device)
                
                noise_pred = noise_pred_uncond + guide_scale * (
                             noise_pred_cond - noise_pred_uncond)

                latent_origin = latent_origin.to(
                    torch.device('cpu') if offload_model else self.device)
                
                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)
                
                if sample_solver == "fm_new":
                    latents_mid_origin = sample_scheduler.step_mid(noise_pred_origin.unsqueeze(0),
                                                                   torch.stack([t]).to(self.device), latent_origin.unsqueeze(0))
                    
                    latents_mid = sample_scheduler.step_mid(noise_pred.unsqueeze(0), 
                                                            torch.stack([t]).to(self.device), latent.unsqueeze(0))
                    
                    latent_model_input_mid = [latents_mid_origin[0].to(self.device),
                                              latents_mid[0].to(self.device), latents_mid[0].to(self.device)]
                    
                    #t_mid = (timestep + sample_scheduler.timesteps[progress_id + 1]) / 2
                    t_mid = (torch.stack([t]) + sample_scheduler.timesteps[progress_id + 1]) / 2
                    timestep_mid = torch.stack([t_mid[0], t_mid[0], t_mid[0]]).to(self.device) 
                    
                    noise_preds_list_mid = self.model(
                        x=latent_model_input_mid,
                        t=timestep_mid,
                        context=context_list,
                        seq_len=max_seq_len,
                        clip_fea=clip_fea,
                        y=y_list,
                        pnp=True,
                        pnp_layers=pnp_layers,
                        progress_id=progress_id, # <--- 新增传递参数
                        sampling_steps=sampling_steps, # <--- 新增传递参数
                        injection_step=injection_step,
                        latent_output_dir=None,
                    ) 
                    
                    if offload_model:
                        torch.cuda.empty_cache()
                    
                    noise_pred_mid_origin = noise_preds_list_mid[0].to(torch.device('cpu') if offload_model else self.device)
                    
                    noise_pred_mid_posi = noise_preds_list_mid[1].to(torch.device('cpu') if offload_model else self.device)
                    noise_pred_mid_nega = noise_preds_list_mid[2].to(torch.device('cpu') if offload_model else self.device)
                    noise_pred_mid = noise_pred_mid_nega + guide_scale * (noise_pred_mid_posi - noise_pred_mid_nega)
                    
                    latent_origin = sample_scheduler.step_solver(noise_pred_mid_origin.unsqueeze(0), noise_pred_origin.unsqueeze(0), 
                                                          torch.stack([t]).to(self.device), latent_origin.unsqueeze(0)).squeeze(0)
                    
                    latent = sample_scheduler.step_solver(noise_pred_mid.unsqueeze(0), noise_pred.unsqueeze(0), 
                                                          torch.stack([t]).to(self.device), latent.unsqueeze(0)).squeeze(0)
                    
                    del latent_model_input_mid
                    
                        
                    
                else:       
                    temp_x0_origin = sample_scheduler.step(
                        noise_pred_origin.unsqueeze(0),
                        t,
                        latent_origin.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent_origin = temp_x0_origin.squeeze(0)
                    
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)

                x0 = [latent_origin.to(self.device), latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latent, latent_origin
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        if self.rank == 0:
            return videos[0], videos[1]
        return None, None

    def generate_reconstruction(self,
                 input_prompt,
                 img,
                 video: torch.Tensor, 
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='fm_new',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 latent_name=None,
                 is_delete=False,
                 ):
        
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        video = video.to(self.device)
        if self.rank == 0:
            print("Reconstruction mode!")

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            round(np.sqrt(max_area * aspect_ratio)) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        # video_latent
        expected_latent_shape = (16, 21, lat_h, lat_w) if not is_delete else (16, 13, lat_h, lat_w)
        latent = self.vae.encode([video])[0]
        if self.rank == 0:
            print(latent.shape)
        assert latent.shape == expected_latent_shape
        
        
        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device) if not is_delete else torch.ones(1, 49, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        
        blank_prompt = ""

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            context_blank = self.text_encoder([blank_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context_blank = self.text_encoder([blank_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            context_blank = [t.to(self.device) for t in context_blank]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                         dim=1).to(self.device)
        ])[0] if not is_delete else self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 48, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        save_intermediate_latents_data = {}

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'fm_new':
                sample_scheduler_inversion = FlowMatchNewScheduler(
                    num_inference_steps=51,
                    num_train_timesteps=self.num_train_timesteps,
                    shift=5.0,
                    inverse_timesteps=True,
                )
                timesteps_inversion = sample_scheduler_inversion.timesteps[:-1]
                if self.rank == 0:
                    print("time_inversion: ", timesteps_inversion)
                sample_scheduler = FlowMatchNewScheduler(
                    num_inference_steps=51,
                    num_train_timesteps=self.num_train_timesteps,
                    shift=5.0,
                )
                timesteps = sample_scheduler.timesteps[:-1]
                if self.rank == 0:
                    print("time: ", timesteps)
                
            else:
                raise NotImplementedError("Unsupported solver in reconstruction mode.")

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            
            # Inversion
            for progress_id, t in enumerate(tqdm(timesteps_inversion, "inversion")):
                
                if progress_id >= 45:
                    save_intermediate_latents_data[t.item()] = latent.clone().cpu()          
                latent_model_input = [latent.to(self.device)]
                
                timestep = torch.stack([t]).to(self.device) 
                context_list = [context_blank[0]]
                clip_fea = torch.cat([clip_context], dim=0)
                y_list = [y]
                
                noise_preds_list = self.model(
                    x=latent_model_input,
                    t=timestep,
                    context=context_list,
                    seq_len=max_seq_len, # seq_len remains the same for padding
                    clip_fea=clip_fea,
                    y=y_list,
                ) 
                
                if offload_model:
                     torch.cuda.empty_cache()
                
                noise_pred = noise_preds_list[0].to(torch.device('cpu') if offload_model else self.device)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)
                
                if sample_solver == "fm_new":
                    latents_mid = sample_scheduler_inversion.step_mid(noise_pred.unsqueeze(0), 
                                                            torch.stack([t]).to(self.device), latent.unsqueeze(0))
                    
                    latent_model_input_mid = [latents_mid[0].to(self.device)]
                    
                    t_mid = (torch.stack([t]) + sample_scheduler_inversion.timesteps[progress_id + 1]) / 2
                    timestep_mid = torch.stack([t_mid[0]]).to(self.device) 
                    
                    noise_preds_list_mid = self.model(
                        x=latent_model_input_mid,
                        t=timestep_mid,
                        context=context_list,
                        seq_len=max_seq_len,
                        clip_fea=clip_fea,
                        y=y_list,
                    ) 
                    
                    if offload_model:
                        torch.cuda.empty_cache()
                    
                    noise_pred_mid = noise_preds_list_mid[0].to(torch.device('cpu') if offload_model else self.device)
                    
                    latent = sample_scheduler_inversion.step_solver(noise_pred_mid.unsqueeze(0), noise_pred.unsqueeze(0), 
                                                          torch.stack([t]).to(self.device), latent.unsqueeze(0)).squeeze(0)
                    del latent_model_input_mid
                    
                else:       
                    temp_x0 = sample_scheduler_inversion.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)
                    

            if self.rank == 0:
                torch.save(save_intermediate_latents_data, latent_name)
                print("Save inversion latent.")
            return None
                    
            
            
            
            
            # Reconstruction
            for progress_id, t in enumerate(tqdm(timesteps)):
                
                latent_model_input = [latent.to(self.device), latent.to(self.device)]
                timestep = torch.stack([t, t]).to(self.device) 
                
                context_list = [context[0], context_null[0]]
                clip_fea = torch.cat([clip_context, clip_context], dim=0)
                y_list = [y, y]
                
                noise_preds_list = self.model(
                    x=latent_model_input,
                    t=timestep,
                    context=context_list,
                    seq_len=max_seq_len, # seq_len remains the same for padding
                    clip_fea=clip_fea,
                    y=y_list,
                ) 
                
                if offload_model:
                     torch.cuda.empty_cache()
                
                noise_pred_cond = noise_preds_list[0].to(torch.device('cpu') if offload_model else self.device)
                noise_pred_uncond = noise_preds_list[1].to(torch.device('cpu') if offload_model else self.device)
                
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)
                
                if sample_solver == "fm_new":
                    latents_mid = sample_scheduler.step_mid(noise_pred.unsqueeze(0), 
                                                            torch.stack([t]).to(self.device), latent.unsqueeze(0))
                    
                    latent_model_input_mid = [latents_mid[0].to(self.device), latents_mid[0].to(self.device)]
                    
                    t_mid = (torch.stack([t]) + sample_scheduler.timesteps[progress_id + 1]) / 2
                    timestep_mid = torch.stack([t_mid[0], t_mid[0]]).to(self.device) 
                    
                    noise_preds_list_mid = self.model(
                        x=latent_model_input_mid,
                        t=timestep_mid,
                        context=context_list,
                        seq_len=max_seq_len,
                        clip_fea=clip_fea,
                        y=y_list,
                    ) 
                    
                    if offload_model:
                        torch.cuda.empty_cache()
                    
                    noise_pred_mid_posi = noise_preds_list_mid[0].to(torch.device('cpu') if offload_model else self.device)
                    noise_pred_mid_nega = noise_preds_list_mid[1].to(torch.device('cpu') if offload_model else self.device)
                    noise_pred_mid = noise_pred_mid_nega + guide_scale * (noise_pred_mid_posi - noise_pred_mid_nega)
                    
                    latent = sample_scheduler.step_solver(noise_pred_mid.unsqueeze(0), noise_pred.unsqueeze(0), 
                                                          torch.stack([t]).to(self.device), latent.unsqueeze(0)).squeeze(0)
                    del latent_model_input_mid
                    
                else:       
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)
        
        del latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None