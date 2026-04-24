import random

import numpy as np
import torch


class FakeScheduler:
    def __init__(self, num_inference_steps, num_train_timesteps):
        self.timesteps = torch.linspace(
            float(num_train_timesteps - 1),
            0.0,
            steps=int(num_inference_steps),
            dtype=torch.float32,
        )

    def add_noise(self, z_src, noise, _t):
        return z_src + 0.05 * noise

    def step(self, eps, _t, sample):
        return sample - 0.1 * eps


class FakeWanModel:
    """
    无卡模型桩：
    - 支持三分支预测（src/tgt/uncond）
    - 支持 ContextFlow 风格 write/read 缓存
    - 通过 pnp_layers 控制是否启用注入
    """

    def __init__(self):
        self.calls = []

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        progress_id,
        sampling_steps,
        pnp=False,
        pnp_layers=None,
        pnp_mode=None,
        pnp_cache=None,
        injection_step=None,
    ):
        self.calls.append(
            {
                "progress_id": int(progress_id),
                "sampling_steps": int(sampling_steps),
                "pnp": bool(pnp),
                "pnp_mode": pnp_mode,
                "pnp_layers": pnp_layers,
                "injection_step": injection_step,
                "batch_size": len(x),
                "seq_len": int(seq_len),
                "t_shape": tuple(t.shape),
                "context_count": len(context),
            }
        )

        outs = []
        for i, latent in enumerate(x):
            branch_bias = 0.02 * (i + 1)
            outs.append(latent.float() * 0.1 + branch_bias)

        if pnp and pnp_mode == "write" and pnp_cache is not None:
            pnp_cache[("ref", int(progress_id))] = outs[0].detach().clone()

        if pnp and pnp_mode == "read" and pnp_cache is not None and pnp_layers:
            ratio = 1.0 if injection_step is None else float(injection_step)
            enable_early = int(progress_id) < max(1, int(ratio * int(sampling_steps)))
            if enable_early and ("ref", int(progress_id)) in pnp_cache and len(outs) >= 2:
                # 仅注入 tgt 分支，模拟“可控注入”
                outs[1] = outs[1] + 0.03 * pnp_cache[("ref", int(progress_id))]

        return outs


class ChordVideoEditorNoGPU:
    """
    自包含无卡版 Chord+PnP 骨架：
    - 保持与你主工程一致的 chord_generate 关键参数
    - 用 FakeWanModel/FakeScheduler 验证流程与开关控制
    """

    def __init__(self):
        self.model = FakeWanModel()
        self.num_train_timesteps = 1000
        self.sample_neg_prompt = "low quality, blur"

    def chord_generate(
        self,
        src_video,
        src_prompt,
        tgt_prompt,
        t_start=0.25,
        step_scale=0.3,
        frame_num=9,
        sampling_steps=8,
        guide_scale=4.5,
        n_prompt="",
        seed=-1,
        max_area=64 * 64,
        **kwargs
    ):
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        sample_solver = kwargs.get("sample_solver", "fm_new")
        use_pnp = kwargs.get("use_pnp", False)
        pnp_layers = kwargs.get("pnp_layers", None)
        pnp_injection_step = kwargs.get("pnp_injection_step", 0.5)
        if sample_solver != "fm_new":
            raise NotImplementedError("无卡测试仅支持 sample_solver='fm_new'")

        if src_video.ndim != 4:
            raise ValueError("src_video 期望形状: (C, F, H, W)")
        if frame_num != int(src_video.shape[1]):
            frame_num = int(src_video.shape[1])
        if int(src_video.shape[2]) * int(src_video.shape[3]) > int(max_area):
            raise ValueError("输入分辨率超过 max_area")

        _ = src_prompt
        _ = tgt_prompt
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        _ = n_prompt

        z_src = src_video.float()
        noise = torch.randn_like(z_src)
        scheduler = FakeScheduler(
            num_inference_steps=sampling_steps,
            num_train_timesteps=self.num_train_timesteps,
        )
        timesteps = scheduler.timesteps
        t_start_value = float(t_start) * float(self.num_train_timesteps)
        start_idx = int(torch.argmin((timesteps - t_start_value).abs()).item())
        timesteps = timesteps[start_idx:]
        z_t = scheduler.add_noise(z_src, noise, timesteps[0])

        latent = z_t.clone()
        latent_ref = z_t.clone()
        pnp_cache = {}
        seq_len = int(frame_num * src_video.shape[2] * src_video.shape[3])

        for progress_id, t in enumerate(timesteps):
            if use_pnp:
                ref_preds = self.model.forward(
                    x=[latent_ref, latent_ref],
                    t=torch.stack([t, t]),
                    context=[torch.zeros(1), torch.zeros(1)],
                    seq_len=seq_len,
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
                    eps_ref_cfg.unsqueeze(0), t, latent_ref.unsqueeze(0)
                ).squeeze(0)

            noise_preds = self.model.forward(
                x=[latent, latent, latent],
                t=torch.stack([t, t, t]),
                context=[torch.zeros(1), torch.zeros(1), torch.zeros(1)],
                seq_len=seq_len,
                progress_id=progress_id,
                sampling_steps=len(timesteps),
                pnp=use_pnp,
                pnp_layers=pnp_layers,
                pnp_mode="read" if use_pnp else None,
                pnp_cache=pnp_cache if use_pnp else None,
                injection_step=pnp_injection_step if use_pnp else None,
            )

            eps_src, eps_tgt, eps_uncond = noise_preds
            eps_src_cfg = eps_uncond + guide_scale * (eps_src - eps_uncond)
            eps_tgt_cfg = eps_uncond + guide_scale * (eps_tgt - eps_uncond)
            chord_direction = eps_tgt_cfg - eps_src_cfg
            eps_edit = eps_src_cfg + step_scale * chord_direction
            latent = scheduler.step(eps_edit.unsqueeze(0), t, latent.unsqueeze(0)).squeeze(0)

        return latent


def run_case(editor, use_pnp):
    src_video = torch.randn(3, 9, 32, 32, dtype=torch.float32)
    return editor.chord_generate(
        src_video=src_video,
        src_prompt="a cat walking on grass",
        tgt_prompt="a robot dog walking on grass",
        t_start=0.25,
        step_scale=0.3,
        frame_num=9,
        sampling_steps=8,
        guide_scale=4.5,
        seed=123,
        sample_solver="fm_new",
        use_pnp=use_pnp,
        pnp_layers=[1, 3, 5],
        pnp_injection_step=0.5,
    )


def assert_result(editor, out_no_pnp, out_with_pnp):
    assert out_no_pnp.shape == (3, 9, 32, 32), "无 PnP 输出形状异常"
    assert out_with_pnp.shape == (3, 9, 32, 32), "有 PnP 输出形状异常"
    assert torch.isfinite(out_no_pnp).all(), "无 PnP 输出存在 NaN/Inf"
    assert torch.isfinite(out_with_pnp).all(), "有 PnP 输出存在 NaN/Inf"
    delta = (out_with_pnp - out_no_pnp).abs().mean().item()
    assert delta > 1e-6, "PnP 开关未生效：输出几乎无差异"
    calls = editor.model.calls
    has_write = any(c["pnp"] and c["pnp_mode"] == "write" for c in calls)
    has_read = any(c["pnp"] and c["pnp_mode"] == "read" for c in calls)
    assert has_write and has_read, "未观察到 PnP write/read 调用链路"


def run_sanity_check_nogpu():
    print("=> 无 GPU 测试开始：自包含 Chord+PnP 骨架")
    editor = ChordVideoEditorNoGPU()
    out_no_pnp = run_case(editor, use_pnp=False)
    out_with_pnp = run_case(editor, use_pnp=True)
    assert_result(editor, out_no_pnp, out_with_pnp)
    print("=> 测试通过：基座稳定 + PnP 可控")
    print(f"no_pnp:  shape={tuple(out_no_pnp.shape)}, mean={out_no_pnp.mean().item():.6f}")
    print(f"with_pnp: shape={tuple(out_with_pnp.shape)}, mean={out_with_pnp.mean().item():.6f}")
    print(f"delta(mean abs)={(out_with_pnp - out_no_pnp).abs().mean().item():.6f}")


if __name__ == "__main__":
    run_sanity_check_nogpu()
