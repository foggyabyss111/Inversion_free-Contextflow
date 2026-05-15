import torch
import logging
import copy
import os
import gc
import numpy as np
import imageio.v2 as imageio
import torch.nn.functional as F
from wan.chord_video_new import ChordVideoEditor
from wan.configs import WAN_CONFIGS

logging.basicConfig(level=logging.INFO)


def load_video_to_tensor(video_path, frame_num=17, height=480, width=832):
    """
    读取视频并转换为 [C, T, H, W] 张量，归一化到 [-1, 1]
    """
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= frame_num:
            break
        frames.append(frame)
    reader.close()

    # 如果帧数不足，循环补齐
    while len(frames) < frame_num:
        frames.append(frames[-1])

    # [T, H, W, C] -> [T, C, H, W]
    video = np.stack(frames)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    
    # 归一化到 [-1, 1]
    video = (video / 127.5) - 1.0
    
    # Resize 到目标分辨率
    video = F.interpolate(video, size=(height, width), mode='bilinear', align_corners=False)
    
    # [T, C, H, W] -> [C, T, H, W]
    video = video.permute(1, 0, 2, 3).contiguous()
    return video


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


def run_real_video_editing():
    model_task = os.getenv("MODEL_TASK", "i2v-14B").strip()
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/root/autodl-tmp/models/Wan2.1-I2V-14B-480P").strip()
    print(f"=> 正在加载模型 ({model_task})...")

    # AutoDL 路径配置
    input_video_path = "/root/Contextflow/ContextFlow/inputs/src.mp4"
    output_dir = "/root/Contextflow/ContextFlow/outputs"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 模型配置选择
    if model_task not in WAN_CONFIGS:
        print(f"!! 错误：不支持的 MODEL_TASK={model_task}，可选值: {sorted(WAN_CONFIGS.keys())}")
        return
    if not os.path.exists(checkpoint_dir):
        print(f"!! 错误：找不到模型目录 {checkpoint_dir}")
        return

    config = copy.deepcopy(WAN_CONFIGS[model_task])
    pipeline = ChordVideoEditor(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        t5_cpu=False,
    )

    print("=> 模型加载完成，准备读取源视频...")
    
    # 低显存安全模式参数（H/W 取 16 的倍数，避免 latent 维度错位）
    frame_num = 17      # 先用 17 帧快速验证
    height = 352
    width = 640
    # SAFE_MODE=1: 稳妥档（更稳更少闪烁）；SAFE_MODE=0: 激进档（更容易改动）
    safe_mode = os.getenv("SAFE_MODE", "0").strip() != "0"
    # 速度优先模式：默认开启。关闭额外对照/重建，并避免模型反复 CPU<->GPU 搬运。
    fast_mode = os.getenv("FAST_MODE", "1").strip() != "0"
    # 可选：仅跑单个 t_start（覆盖默认与扫描）。示例：export SINGLE_T_START=0.82
    single_t_start = os.getenv("SINGLE_T_START", "").strip()
    # 可选：开启 base t_start 单变量扫描。示例：export RUN_BASE_SWEEP=1
    run_base_sweep = os.getenv("RUN_BASE_SWEEP", "0").strip() == "1"
    # 可选：编辑策略预设。EDIT_PROFILE=fidelity(保真优先) 或 color(改色优先)
    edit_profile = os.getenv("EDIT_PROFILE", "color").strip().lower()
    # baseline 的 inversion / edit 步数与注入参数
    inversion_steps = int(os.getenv("INVERSION_STEPS", "50").strip())
    edit_steps = int(os.getenv("EDIT_STEPS", "40").strip())
    injection_step = float(os.getenv("INJECTION_STEP", "0.5").strip())
    save_last_n_steps = int(os.getenv("SAVE_LAST_N_STEPS", "8").strip())

    # baseline 下可选保留一轮 reconstruction sanity。
    run_mid_recon = os.getenv("RUN_MID_RECON", "0").strip() == "1"
    # 跳过第零轮重建，直接测 base（速度优先）。
    skip_recon = os.getenv("SKIP_RECON", "0").strip() == "1"
    if fast_mode:
        run_base_sweep = False
        run_mid_recon = False
        skip_recon = True
    if edit_profile not in {"fidelity", "color"}:
        print(f"=> 未知 EDIT_PROFILE={edit_profile}，自动回退到 fidelity")
        edit_profile = "fidelity"
    
    if not os.path.exists(input_video_path):
        print(f"!! 错误：找不到输入视频 {input_video_path}，请先上传视频。")
        return

    src_video = load_video_to_tensor(input_video_path, frame_num=frame_num, height=height, width=width)
    src_video = src_video.to("cuda", dtype=torch.bfloat16)
    
    # 提示词设置（先用简洁描述，避免整图风格染色）
    source_prompt = "A white ceramic mug on a wooden table, natural lighting, high quality."
    target_prompt = "A black ceramic mug on a wooden table, natural lighting, high quality."

    print(f"=> 开始编辑: '{source_prompt}' -> '{target_prompt}'")
    print(f"=> SAFE_MODE={'ON(稳妥)' if safe_mode else 'OFF(激进)'}")
    print(f"=> FAST_MODE={'ON(速度优先)' if fast_mode else 'OFF(完整实验)'}")
    print(f"=> RUN_BASE_SWEEP={'ON' if run_base_sweep else 'OFF'}")
    print(f"=> RUN_MID_RECON={'ON' if run_mid_recon else 'OFF'}")
    print(f"=> SKIP_RECON={'ON' if skip_recon else 'OFF'}")
    print(f"=> MODEL_TASK={model_task}")
    print(f"=> CHECKPOINT_DIR={checkpoint_dir}")
    print(f"=> INVERSION_STEPS={inversion_steps}")
    print(f"=> EDIT_STEPS={edit_steps}")
    print(f"=> SAVE_LAST_N_STEPS={save_last_n_steps}")
    print(f"=> INJECTION_STEP={injection_step}")
    print(f"=> EDIT_PROFILE={edit_profile}")
    
    try:
        if not skip_recon:
            # 第零轮：重建 sanity（同 prompt，极低 t_start，确认链路先能出干净视频）
            print("=> 开始第零轮：极低噪声重建测试...")
            recon_result = pipeline.chord_generate(
                src_video=src_video,
                src_prompt=source_prompt,
                tgt_prompt=source_prompt,
                t_start=0.65,
                frame_num=frame_num,
                sampling_steps=edit_steps,
                guide_scale=4.5,      # 较低引导
                sample_solver="fm_new",
                injection_step=injection_step,
                inversion_steps=inversion_steps,
                edit_steps=edit_steps,
                save_last_n_steps=save_last_n_steps,
                offload_model=not fast_mode,
            )
            recon_file = os.path.join(output_dir, "chord_recon_sanity.mp4")
            save_video_tensor_to_mp4(recon_result, recon_file, fps=8)
            print(f"=> [0/x] 重建 sanity 视频已保存: {recon_file}")
            del recon_result
            if not fast_mode:
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print("=> 已跳过第零轮重建，直接进入 base 编辑。")

        # 第一轮：基座编辑
        if edit_profile == "fidelity":
            base_t_start = 0.68
            base_sampling_steps = edit_steps
            base_guide_scale = 4.5
        else:
            base_t_start = 0.82
            base_sampling_steps = edit_steps
            base_guide_scale = 5.5

        if safe_mode:
            base_t_start = max(0.65, base_t_start - 0.05)
            base_guide_scale = max(4.2, base_guide_scale - 0.4)
            base_sampling_steps = min(56, base_sampling_steps + 4)

        if run_mid_recon:
            # 同 prompt 的 baseline reconstruction sanity。
            print(f"=> 开始第 0.5 轮：在 t_start={base_t_start} 下的重建测试...")
            recon_edit_t = pipeline.chord_generate(
                src_video=src_video,
                src_prompt=source_prompt,
                tgt_prompt=source_prompt,
                t_start=base_t_start,
                frame_num=frame_num,
                sampling_steps=base_sampling_steps,
                guide_scale=base_guide_scale,
                sample_solver="fm_new",
                injection_step=injection_step,
                inversion_steps=inversion_steps,
                edit_steps=base_sampling_steps,
                save_last_n_steps=save_last_n_steps,
                offload_model=not fast_mode,
            )
            recon_edit_t_file = os.path.join(output_dir, f"chord_recon_tstart_{base_t_start:.2f}.mp4")
            save_video_tensor_to_mp4(recon_edit_t, recon_edit_t_file, fps=8)
            print(f"=> [0.5/x] 重建验证视频已保存: {recon_edit_t_file}")
            del recon_edit_t
            if not fast_mode:
                gc.collect()
                torch.cuda.empty_cache()

        if single_t_start:
            t_start_list = [float(single_t_start)]
        elif run_base_sweep:
            t_start_list = [
                max(0.65, base_t_start - 0.05),
                base_t_start,
                min(0.90, base_t_start + 0.05),
            ]
        else:
            t_start_list = [base_t_start]

        print(f"=> 开始 Base 文本编辑，t_start 列表: {t_start_list}")
        for idx, cur_t_start in enumerate(t_start_list, start=1):
            print(f"=> [1.{idx}] 运行 base 版本, t_start={cur_t_start:.2f}")
            result = pipeline.chord_generate(
                src_video=src_video,
                src_prompt=source_prompt,
                tgt_prompt=target_prompt,
                t_start=cur_t_start,
                frame_num=frame_num,
                sampling_steps=base_sampling_steps,
                guide_scale=base_guide_scale,
                sample_solver="fm_new",
                injection_step=injection_step,
                inversion_steps=inversion_steps,
                edit_steps=base_sampling_steps,
                save_last_n_steps=save_last_n_steps,
                latent_cache_dir=os.path.join(output_dir, "baseline_latents"),
                offload_model=not fast_mode,
            )

            if len(t_start_list) == 1:
                out_file = os.path.join(output_dir, "chord_base_edit.mp4")
            else:
                out_file = os.path.join(
                    output_dir,
                    f"chord_base_edit_tstart_{cur_t_start:.2f}.mp4"
                )
            save_video_tensor_to_mp4(result, out_file, fps=8)
            print(f"=>     已保存: {out_file}")
            del result
            if not fast_mode:
                gc.collect()
                torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"=> 运行时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_real_video_editing()
