import torch
import logging
import copy
import os
import numpy as np
import imageio.v2 as imageio
from wan.chord_video_new import ChordVideoEditor
from wan.configs import WAN_CONFIGS

logging.basicConfig(level=logging.INFO)


def save_video_tensor_to_mp4(video_tensor, out_path, fps=8):
    """
    video_tensor: torch.Tensor, shape [C, T, H, W], value range roughly in [-1, 1].
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    video = video_tensor.detach().float().cpu().clamp(-1, 1)
    video = ((video + 1.0) * 127.5).to(torch.uint8)  # [C, T, H, W]
    video = video.permute(1, 2, 3, 0).contiguous().numpy()  # [T, H, W, C]
    with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
        for frame in video:
            writer.append_data(np.asarray(frame))


def run_sanity_check():
    print("=> 正在加载模型...")
    
    # 先查看有哪些可用配置（调试用）
    print("可用配置:", [k for k in WAN_CONFIGS.keys() if 't2v' in k.lower() or '1.3' in k])
    
    # 根据实际键名选择（常见的是 't2v-1_3B' 或 't2v-14B'）
    # 如果你下的是 1.3B，用 't2v-1.3B'；如果下的是 14B，用 't2v-14B'
    config = copy.deepcopy(WAN_CONFIGS["t2v-1.3B"])
    for key in ["clip_model", "clip_dtype", "clip_checkpoint", "clip_tokenizer"]:
        if hasattr(config, key):
            delattr(config, key)
    
    pipeline = ChordVideoEditor(
        config=config,
        checkpoint_dir="/root/autodl-tmp/models/Wan2.1-T2V-1.3B",
        device_id=0,
        t5_cpu=False,
    )

    print("=> 模型加载完成，开始测试...")
    
    source_prompt = "A cat walking on the grass, close up, high quality."
    target_prompt = "A robot dog walking on the grass, metallic texture, close up."

    # 创建假视频用于测试
    # 注意：根据你的实际需求调整视频张量的形状
    dummy_video = torch.randn(3, 17, 480, 832, dtype=torch.bfloat16, device="cuda")
    
    try:
        result = pipeline.chord_generate(
            src_video=dummy_video,
            src_prompt=source_prompt,
            tgt_prompt=target_prompt,
            t_start=0.8,
            step_scale=1.0,
            frame_num=81,
            sampling_steps=20,
            guide_scale=5.0,
            sample_solver="fm_new",
            use_pnp=False,
        )
        print(f"=> 测试跑通！输出形状: {result.shape}")
        out_file = "/root/Contextflow/ContextFlow/outputs/chord_sanity.mp4"
        save_video_tensor_to_mp4(result, out_file, fps=8)
        print(f"=> 视频已保存: {out_file}")
        
    except Exception as e:
        print(f"=> 出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_sanity_check()
