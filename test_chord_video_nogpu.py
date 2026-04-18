import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NoGpuTestConfig:
    """最小配置，仅用于无卡逻辑打点。"""

    sample_neg_prompt: str = ""


class ChordVideoEditorNoGPU:
    """
    无卡/低资源环境下的轻量测试版 Editor。
    - 不加载 WAN/T5/VAE/CLIP 权重
    - 保留 chord_generate 的核心接口形态
    - 用可控的“伪编辑”模拟语义改动流程
    """

    def __init__(self, config: NoGpuTestConfig):
        self.config = config
        self.device = torch.device("cpu")

    def chord_generate(
        self,
        src_video,  # (C, N, H, W)
        src_prompt,
        tgt_prompt,
        t_start=0.9,
        step_scale=0.5,
        frame_num=81,
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        max_area=720 * 1280,
        **kwargs
    ):
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if src_video.ndim != 4:
            raise ValueError(
                f"src_video 期望形状 (C, N, H, W)，实际得到 {tuple(src_video.shape)}"
            )

        c, n, h, w = src_video.shape
        if c != 3:
            raise ValueError(f"当前测试脚本仅支持 RGB 视频，实际 C={c}")
        if n < 2:
            raise ValueError("帧数至少为 2，便于验证时序逻辑。")
        if h * w > max_area:
            raise ValueError(
                f"分辨率超过 max_area: {h}x{w}={h*w} > {max_area}"
            )

        # 模拟“源/目标语义方向”：根据 prompt 差异构造一个标量强度
        semantic_delta = (len(tgt_prompt) - len(src_prompt)) * 1e-3
        control = float(step_scale) * float(guide_scale) * (1.0 - float(t_start))

        # 可复现噪声，代表扩散步中的扰动项
        noise = torch.randn_like(src_video, dtype=torch.float32) * 0.01

        # 伪编辑：保留主体 + 少量语义偏移 + 微弱噪声
        edited = src_video.float() + control * semantic_delta + noise
        edited = torch.clamp(edited, -3.0, 3.0)
        return edited


def run_sanity_check_nogpu():
    print("=> 无卡 Sanity Check（不加载大模型）开始...")

    config = NoGpuTestConfig(
        sample_neg_prompt="low quality, blurry, artifacts"
    )
    pipeline = ChordVideoEditorNoGPU(config=config)

    source_prompt = "A cat walking on the grass, close up, high quality."
    target_prompt = "A robot dog walking on the grass, metallic texture, close up."

    # 无卡场景建议小尺寸，保证稳定
    dummy_video = torch.randn(3, 9, 144, 256, dtype=torch.float32, device="cpu")

    out = pipeline.chord_generate(
        src_video=dummy_video,
        src_prompt=source_prompt,
        tgt_prompt=target_prompt,
        t_start=0.8,
        step_scale=1.0,
        frame_num=9,
        sampling_steps=20,
        guide_scale=5.0,
        seed=42,
        max_area=256 * 256,
    )

    print("=> 跑通成功")
    print(f"输入形状: {tuple(dummy_video.shape)}")
    print(f"输出形状: {tuple(out.shape)}")
    print(f"输出统计: mean={out.mean().item():.6f}, std={out.std().item():.6f}")


if __name__ == "__main__":
    run_sanity_check_nogpu()
