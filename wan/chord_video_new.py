# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Chord Video Editing - 基于ChordEdit算法的视频编辑
核心逻辑：在扩散过程中通过双提示词预测计算编辑方向
当前版本面向Wan2.1 T2V路径，仅使用文本条件
"""

# 设置OpenMP环境变量以避免libgomp错误
import os
os.environ['OMP_NUM_THREADS'] = '1'

import gc
import logging
import math
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
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers_modified import FlowMatchNewScheduler


class ChordVideoEditor:
    """
    Chord视频编辑类，基于ChordEdit算法实现视频语义编辑
    核心思想：在扩散过程中通过双提示词预测计算编辑方向，然后沿着这个方向更新特征
    当前版本面向Wan2.1 T2V路径，仅使用文本条件
    """

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
        初始化Chord视频编辑模型组件

        Args:
            config (EasyDict): 模型配置参数
            checkpoint_dir (str): 检查点目录路径
            device_id (int, 可选, 默认0): GPU设备ID
            rank (int, 可选, 默认0): 分布式训练进程排名
            t5_fsdp (bool, 可选, 默认False): T5模型FSDP分片
            dit_fsdp (bool, 可选, 默认False): DiT模型FSDP分片
            use_usp (bool, 可选, 默认False): 启用USP分布策略
            t5_cpu (bool, 可选, 默认False): T5模型放在CPU上
            init_on_cpu (bool, 可选, 默认True): 在CPU上初始化Transformer模型
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        
        # 初始化文本编码器
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # VAE参数
        self.vae_stride = config.vae_stride  # (4, 8, 8) - 时间下采样4倍，空间下采样8倍
        self.patch_size = config.patch_size  # (1, 2, 2) - 时间不分割，空间分割为2x2块
        
        # 初始化VAE
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # 加载主模型
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        # USP序列并行支持
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        # 分布式训练同步
        if dist.is_initialized():
            dist.barrier()
        
        # 模型分片或设备移动
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        # 默认负提示词
        self.sample_neg_prompt = config.sample_neg_prompt

    def chord_generate(
        self,
        src_video,                    # 源视频张量 (C, N, H, W)
        src_prompt,                   # 源视频描述
        tgt_prompt,                   # 目标视频描述
        t_start=0.9,                  # 噪声起点时间步 (0-1)
        step_scale=0.5,               # 编辑步长缩放因子
        frame_num=81,                 # 帧数
        sampling_steps=50,            # 采样步数
        guide_scale=5.0,              # 分类器自由引导尺度
        n_prompt="",                  # 负提示词
        seed=-1,                      # 随机种子
        offload_model=True,           # 模型卸载
        max_area=720 * 1280,          # 最大像素面积
        **kwargs
    ):
        """
        执行Chord视频编辑
        
        Args:
            src_video: 源视频张量 (C, N, H, W)
            src_prompt: 源视频描述
            tgt_prompt: 目标视频描述
            t_start: 噪声起点时间步 (0-1)
            step_scale: 编辑步长缩放因子
            frame_num: 帧数
            sampling_steps: 采样步数
            guide_scale: 分类器自由引导尺度
            n_prompt: 负提示词
            seed: 随机种子
            offload_model: 模型卸载
            max_area: 最大像素面积
            **kwargs: 其他参数
            
        Returns:
            编辑后的视频张量
        """
        # 设置随机种子
        if seed >= 0:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        shift = kwargs.get("shift", 5.0)
        sample_solver = kwargs.get("sample_solver", "fm_new")
        use_pnp = kwargs.get("use_pnp", False)
        pnp_layers = kwargs.get("pnp_layers", None)
        pnp_injection_step = kwargs.get("pnp_injection_step", 0.2)

        if sample_solver != "fm_new":
            raise NotImplementedError(
                f"当前最小骨架仅支持 sample_solver='fm_new'，收到: {sample_solver}"
            )

        # 规范化输入为 [C, F, H, W]
        if src_video.ndim != 4:
            raise ValueError(
                f"src_video 期望 4 维 (C, F, H, W)，实际形状: {tuple(src_video.shape)}"
            )
        src_video = src_video.to(self.device, dtype=torch.float32)
        frame_num_from_video = int(src_video.shape[1])
        if frame_num != frame_num_from_video:
            logging.warning(
                "frame_num(%s) 与 src_video 帧数(%s)不一致，使用 src_video 实际帧数。",
                frame_num,
                frame_num_from_video,
            )
            frame_num = frame_num_from_video

        # latent 形状与序列长度
        lat_h = src_video.shape[2] // self.vae_stride[1]
        lat_w = src_video.shape[3] // self.vae_stride[2]
        if lat_h * lat_w > max_area // (self.vae_stride[1] * self.vae_stride[2]):
            logging.warning("输入分辨率较高，可能导致显存不足。")

        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # 文本条件
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_src = self.text_encoder([src_prompt], self.device)
            context_tgt = self.text_encoder([tgt_prompt], self.device)
            context_neg = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_src = self.text_encoder([src_prompt], torch.device("cpu"))
            context_tgt = self.text_encoder([tgt_prompt], torch.device("cpu"))
            context_neg = self.text_encoder([n_prompt], torch.device("cpu"))
            context_src = [t.to(self.device) for t in context_src]
            context_tgt = [t.to(self.device) for t in context_tgt]
            context_neg = [t.to(self.device) for t in context_neg]

        # 源视频编码到 latent 空间
        z_src = self.vae.encode([src_video])[0].to(self.device, dtype=torch.float32)
        noise = torch.randn_like(z_src, dtype=torch.float32, device=self.device)

        # 构造采样器并按 t_start 加噪得到起点 z_t
        scheduler = FlowMatchNewScheduler(
            num_inference_steps=sampling_steps,
            num_train_timesteps=self.num_train_timesteps,
            shift=shift,
        )
        timesteps = scheduler.timesteps
        t_start_value = float(t_start) * float(self.num_train_timesteps)
        t_start_tensor = torch.tensor(t_start_value, dtype=timesteps.dtype)
        start_idx = int(torch.argmin((timesteps - t_start_tensor).abs()).item())
        timesteps = timesteps[start_idx:]
        z_t = scheduler.add_noise(z_src, noise, timesteps[0])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            self.model.to(self.device)
            latent = z_t
            latent_ref = z_t.clone()
            pnp_cache = {}

            # 三分支（src/tgt/uncond）+ Chord 方向编辑
            for progress_id, t in enumerate(tqdm(timesteps, desc="Chord sampling")):
                if use_pnp:
                    # write: 用 source/reference 分支写入 K/V cache
                    ref_preds = self.model(
                        x=[latent_ref, latent_ref],
                        t=torch.stack([t, t]).to(self.device),
                        context=[context_src[0], context_neg[0]],
                        seq_len=max_seq_len,
                        progress_id=progress_id,
                        sampling_steps=len(timesteps),
                        pnp=True,
                        pnp_layers=pnp_layers,
                        pnp_mode="write",
                        pnp_cache=pnp_cache,
                        injection_step=pnp_injection_step,
                    )
                    eps_ref_cfg = ref_preds[1] + guide_scale * (ref_preds[0] - ref_preds[1])
                    latent_ref = scheduler.step(
                        eps_ref_cfg.unsqueeze(0),
                        t,
                        latent_ref.unsqueeze(0),
                    ).squeeze(0)

                noise_preds = self.model(
                    x=[latent, latent, latent],
                    t=torch.stack([t, t, t]).to(self.device),
                    context=[context_src[0], context_tgt[0], context_neg[0]],
                    seq_len=max_seq_len,
                    progress_id=progress_id,
                    sampling_steps=len(timesteps),
                    pnp=use_pnp,
                    pnp_layers=pnp_layers,
                    pnp_mode="read" if use_pnp else None,
                    pnp_cache=pnp_cache if use_pnp else None,
                    injection_step=pnp_injection_step if use_pnp else None,
                )

                eps_src = noise_preds[0]
                eps_tgt = noise_preds[1]
                eps_uncond = noise_preds[2]

                # 分别做 CFG，再用 Chord residual 做方向更新
                eps_src_cfg = eps_uncond + guide_scale * (eps_src - eps_uncond)
                eps_tgt_cfg = eps_uncond + guide_scale * (eps_tgt - eps_uncond)
                chord_direction = eps_tgt_cfg - eps_src_cfg
                eps_edit = eps_src_cfg + step_scale * chord_direction

                latent = scheduler.step(
                    eps_edit.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                ).squeeze(0)

            videos = self.vae.decode([latent.to(self.device)])

            if offload_model:
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del z_src, noise, z_t, latent, latent_ref, scheduler
        if offload_model and torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0]

    def _encode_text(self, prompt):
        """编码文本提示词"""
        return self.text_encoder.encode([prompt])
