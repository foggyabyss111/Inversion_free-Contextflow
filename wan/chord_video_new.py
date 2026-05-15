# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Chord Video Editing baseline wrapper.

This module intentionally stops acting as a standalone solver. Instead it
orchestrates the stable baseline path:
1. video -> inversion latents
2. latent_origin + latent_edit in WanI2V
3. model.py injection + joint solver stepping
4. decode edited video
"""

import gc
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .image2video import WanI2V


class ChordVideoEditor:
    """Thin wrapper that preserves the ChordVideoEditor interface.

    The current mainline baseline is inversion-grounded. This class simply
    prepares inputs, runs inversion, selects a latent timestep, and delegates
    the actual edit to WanI2V.
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
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.rank = rank
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.device = torch.device(f"cuda:{device_id}")
        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_train_timesteps = config.num_train_timesteps

        self.pipeline = WanI2V(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            t5_cpu=t5_cpu,
            init_on_cpu=init_on_cpu,
        )

    def _video_first_frame_to_pil(self, src_video: torch.Tensor) -> Image.Image:
        if src_video.ndim != 4:
            raise ValueError(
                f"src_video expected shape [C, F, H, W], got {tuple(src_video.shape)}"
            )
        frame = src_video[:, 0].detach().float().cpu().clamp(-1, 1)
        frame = ((frame + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(frame)

    def _load_latent_cache(self, latent_path):
        latent_cache = torch.load(latent_path, map_location="cpu")
        if not latent_cache:
            raise ValueError(f"Inversion latent file is empty: {latent_path}")
        return latent_cache

    def _pick_latent_t(self, latent_cache, t_start):
        target_t = float(max(0.0, min(1.0, t_start)) * self.num_train_timesteps)
        available = sorted(float(k) for k in latent_cache.keys())
        return min(available, key=lambda t: abs(t - target_t))

    def chord_generate(
        self,
        src_video,
        src_prompt,
        tgt_prompt,
        t_start=0.8,
        step_scale=0.5,
        frame_num=81,
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        max_area=720 * 1280,
        anchor_keep_early=0.8,
        anchor_keep_late=0.2,
        anchor_noise_seed=42,
        **kwargs,
    ):
        """Run the inversion-grounded baseline edit.

        Notes:
        - `step_scale` and old anchor-related knobs are no longer active in the
          baseline path. Editing strength is mainly controlled by `t_start`,
          `guide_scale`, `injection_step`, and the inversion timestep selected.
        - The first frame of `src_video` is reused as both `img` and
          `img_origin`, which matches the current image-to-video interface.
        """
        del step_scale, anchor_keep_early, anchor_keep_late, anchor_noise_seed

        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if src_video.ndim != 4:
            raise ValueError(
                f"src_video expected shape [C, F, H, W], got {tuple(src_video.shape)}"
            )

        actual_frame_num = int(src_video.shape[1])
        if frame_num != actual_frame_num:
            logging.warning(
                "frame_num(%s) != src_video frames(%s); using the actual frame count.",
                frame_num,
                actual_frame_num,
            )
            frame_num = actual_frame_num

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        shift = float(kwargs.get("shift", 5.0))
        sample_solver = kwargs.get("sample_solver", "fm_new")
        inversion_steps = int(max(4, kwargs.get("inversion_steps", max(sampling_steps, 50))))
        edit_steps = int(max(4, kwargs.get("edit_steps", sampling_steps)))
        injection_step = float(kwargs.get("injection_step", 0.5))
        pnp_layers = kwargs.get("pnp_layers", [27, 29, 31, 32])
        save_last_n_steps = int(max(1, kwargs.get("save_last_n_steps", 8)))
        latent_output_dir = kwargs.get("latent_output_dir", None)
        load_intermediate_latent_path = kwargs.get("load_intermediate_latent_path", None)
        load_intermediate_latent_t = kwargs.get("load_intermediate_latent_t", None)
        save_inversion_latent_path = kwargs.get("save_inversion_latent_path", None)
        latent_cache_dir = kwargs.get(
            "latent_cache_dir",
            os.path.join(os.getcwd(), "outputs", "baseline_latents"),
        )

        deprecated_knobs = [
            "chord_t_delta",
            "u_hat_clip_norm",
            "edit_update_mode",
            "direction_mode",
            "direction_cfg_weight",
            "direction_cfg_scale",
            "n_steps",
            "t_end",
            "trajectory_seed",
            "run_inversion",
        ]
        deprecated_used = [k for k in deprecated_knobs if k in kwargs]
        if deprecated_used:
            logging.warning("Baseline path ignores experimental knobs: %s", ", ".join(deprecated_used))

        if sample_solver != "fm_new":
            raise NotImplementedError(
                f"Baseline path currently only supports sample_solver='fm_new', got: {sample_solver}"
            )

        src_video = src_video.detach().to(self.device, dtype=torch.float32)
        src_img = self._video_first_frame_to_pil(src_video)

        auto_latent = load_intermediate_latent_path is None
        if auto_latent:
            Path(latent_cache_dir).mkdir(parents=True, exist_ok=True)
            latent_path = save_inversion_latent_path
            if latent_path is None:
                seed_tag = str(seed) if seed >= 0 else "auto"
                latent_path = os.path.join(
                    latent_cache_dir,
                    f"chord_baseline_seed_{seed_tag}_f{frame_num}_s{inversion_steps}.pt",
                )
            latent_cache = self.pipeline.run_inversion(
                input_prompt=src_prompt,
                img=src_img,
                video=src_video,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver="fm_new",
                sampling_steps=inversion_steps,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                latent_name=latent_path,
                is_delete=False,
                save_last_n_steps=save_last_n_steps,
            )
            if latent_cache is None:
                latent_cache = self._load_latent_cache(latent_path)
        else:
            latent_path = load_intermediate_latent_path
            latent_cache = self._load_latent_cache(latent_path)

        selected_t = (
            float(load_intermediate_latent_t)
            if load_intermediate_latent_t is not None
            else self._pick_latent_t(latent_cache, t_start)
        )
        logging.info("Baseline edit uses inversion latent t=%.4f", selected_t)

        _, edited_video = self.pipeline.run_edit_from_inversion(
            input_prompt=tgt_prompt,
            input_prompt_origin=src_prompt,
            img=src_img,
            img_origin=src_img,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver="fm_new",
            sampling_steps=edit_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
            pnp_layers=pnp_layers,
            load_intermediate_latent_path=latent_path,
            load_intermediate_latent_t=selected_t,
            injection_step=injection_step,
            latent_output_dir=latent_output_dir,
            is_delete=False,
        )

        cleanup_inversion_latent = bool(kwargs.get("cleanup_inversion_latent", auto_latent and save_inversion_latent_path is None))
        if cleanup_inversion_latent and auto_latent and os.path.exists(latent_path):
            try:
                os.remove(latent_path)
            except OSError:
                logging.warning("Failed to remove temporary inversion latent: %s", latent_path)

        if offload_model and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        return edited_video
