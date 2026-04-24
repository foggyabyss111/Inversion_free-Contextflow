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
    print("=> 正在加载模型 (Wan2.1-T2V-1.3B)...")
    
    # AutoDL 路径配置
    checkpoint_dir = "/root/autodl-tmp/models/Wan2.1-T2V-1.3B"
    input_video_path = "/root/Contextflow/ContextFlow/inputs/src.mp4"
    output_dir = "/root/Contextflow/ContextFlow/outputs"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 模型配置选择
    config = copy.deepcopy(WAN_CONFIGS["t2v-1.3B"])
    # 移除不需要的 clip 相关配置（如果存在）
    for key in ["clip_model", "clip_dtype", "clip_checkpoint", "clip_tokenizer"]:
        if hasattr(config, key):
            delattr(config, key)
    
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
    # 可选：仅跑单个 step_scale（覆盖默认与扫描）。示例：export SINGLE_STEP_SCALE=0.35
    single_step_scale = os.getenv("SINGLE_STEP_SCALE", "").strip()
    # 可选：开启 base step_scale 单变量扫描。示例：export RUN_BASE_SWEEP=1
    run_base_sweep = os.getenv("RUN_BASE_SWEEP", "0").strip() == "1"
    # A/B 对照：legacy_eps(旧路径) 与 u_hat(新路径)。默认开启，便于一次命令定位问题来源。
    run_ab_compare = os.getenv("RUN_AB_COMPARE", "1").strip() != "0"
    # 可选：编辑策略预设。EDIT_PROFILE=fidelity(保真优先) 或 color(改色优先)
    edit_profile = os.getenv("EDIT_PROFILE", "color").strip().lower()
    # 方向估计模式：
    # no_cfg: 最稳但改色偏弱；cfg: 改色强但风险高；mix: 推荐平衡档。
    direction_mode = os.getenv("DIRECTION_MODE", "mix").strip().lower()
    direction_cfg_weight = float(os.getenv("DIRECTION_CFG_WEIGHT", "0.45").strip())
    direction_cfg_scale = float(os.getenv("DIRECTION_CFG_SCALE", "5.6").strip())
    # Inversion-Free 版本下该检查默认关闭（step_scale=0 时与第零轮等价）。
    run_mid_recon = os.getenv("RUN_MID_RECON", "0").strip() == "1"
    # 跳过第零轮重建，直接测 base（速度优先）。
    skip_recon = os.getenv("SKIP_RECON", "0").strip() == "1"
    if fast_mode:
        run_base_sweep = False
        run_ab_compare = False
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
    print(f"=> RUN_AB_COMPARE={'ON' if run_ab_compare else 'OFF'}")
    print(f"=> RUN_MID_RECON={'ON' if run_mid_recon else 'OFF'}")
    print(f"=> SKIP_RECON={'ON' if skip_recon else 'OFF'}")
    print(
        "=> DIRECTION: "
        f"mode={direction_mode}, cfg_weight={direction_cfg_weight}, cfg_scale={direction_cfg_scale}"
    )
    print(f"=> EDIT_PROFILE={edit_profile}")
    
    try:
        if not skip_recon:
            # 第零轮：重建 sanity（同 prompt，极低 t_start，确认链路先能出干净视频）
            print("=> 开始第零轮：极低噪声重建测试...")
            recon_result = pipeline.chord_generate(
                src_video=src_video,
                src_prompt=source_prompt,
                tgt_prompt=source_prompt,
                t_start=0.10,         # 极低噪声起点
                step_scale=0.0,       # 零编辑强度
                frame_num=frame_num,
                sampling_steps=40,    # 充足步数
                guide_scale=4.5,      # 较低引导
                sample_solver="fm_new",
                use_pnp=False,
                min_effective_steps=0,
                chord_t_delta=0.04,
                u_hat_clip_norm=0.0,
                offload_model=not fast_mode,
            )
            recon_file = os.path.join(output_dir, "chord_recon_sanity_t0.1.mp4")
            save_video_tensor_to_mp4(recon_result, recon_file, fps=8)
            print(f"=> [0/x] 重建 sanity 视频已保存: {recon_file}")
            del recon_result
            if not fast_mode:
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print("=> 已跳过第零轮重建，直接进入 base 编辑。")

        # 第一轮：基座编辑 (use_pnp=False)
        if edit_profile == "fidelity":
            base_t_start = 0.70
            base_step_scale = 0.06
            base_sampling_steps = 48
            base_guide_scale = 4.5
            base_min_effective_steps = 36
            base_n_steps = 24
            base_t_end = 0.25
        else:
            # 撤销 cleanup 错误尝试，从参数层面根本性压制首帧马赛克和边缘闪烁
            base_t_start = 0.72  # 从 0.78 进一步降至 0.72，彻底避开高噪区崩溃点
            base_step_scale = 0.08  # 同步收紧步长，减少闪烁
            base_sampling_steps = 56
            base_guide_scale = 5.0
            base_min_effective_steps = 42
            base_n_steps = 32  # 继续增加迭代步数，让过渡极其平滑
            base_t_end = 0.20

        if safe_mode:
            base_t_start = max(0.78, base_t_start - 0.05)
            base_step_scale = max(0.08, base_step_scale - 0.02)
            base_guide_scale = max(4.2, base_guide_scale - 0.3)
            base_sampling_steps = min(56, base_sampling_steps + 4)
            base_min_effective_steps = min(base_sampling_steps, base_min_effective_steps + 2)

        if run_mid_recon:
            # 仅在显式开启时运行；Inversion-Free 下默认可跳过
            print(f"=> 开始第 0.5 轮：在 t_start={base_t_start} 下的重建测试...")
            recon_edit_t = pipeline.chord_generate(
                src_video=src_video,
                src_prompt=source_prompt,
                tgt_prompt=source_prompt,
                t_start=base_t_start,
                step_scale=0.0,
                frame_num=frame_num,
                sampling_steps=base_sampling_steps,
                guide_scale=base_guide_scale,
                sample_solver="fm_new",
                use_pnp=False,
                min_effective_steps=base_min_effective_steps,
                offload_model=not fast_mode,
            )
            recon_edit_t_file = os.path.join(output_dir, f"chord_recon_t{base_t_start:.2f}.mp4")
            save_video_tensor_to_mp4(recon_edit_t, recon_edit_t_file, fps=8)
            print(f"=> [0.5/x] 重建验证视频已保存: {recon_edit_t_file}")
            del recon_edit_t
            if not fast_mode:
                gc.collect()
                torch.cuda.empty_cache()

        if single_step_scale:
            step_scale_list = [float(single_step_scale)]
        elif run_base_sweep:
            step_scale_list = [
                max(0.02, base_step_scale - 0.02),
                base_step_scale,
                min(0.20, base_step_scale + 0.05),
            ]
        else:
            step_scale_list = [base_step_scale]

        print(f"=> 开始 Base 文本编辑，step_scale 列表: {step_scale_list}")
        for idx, cur_step_scale in enumerate(step_scale_list, start=1):
            print(f"=> [1.{idx}] 运行 base 版本, step_scale={cur_step_scale:.2f}")
            mode_list = ["legacy_eps", "u_hat"] if run_ab_compare else ["u_hat"]
            for mode in mode_list:
                print(f"=>       模式={mode}")
                result = pipeline.chord_generate(
                    src_video=src_video,
                    src_prompt=source_prompt,
                    tgt_prompt=target_prompt,
                    t_start=base_t_start,
                    step_scale=cur_step_scale,
                    frame_num=frame_num,
                    sampling_steps=base_sampling_steps,
                    guide_scale=base_guide_scale,
                    sample_solver="fm_new",
                    use_pnp=False,
                    min_effective_steps=base_min_effective_steps,
                    chord_t_delta=0.04,
                    u_hat_clip_norm=0.0,
                    edit_update_mode=mode,
                    direction_mode=direction_mode,
                    direction_cfg_weight=direction_cfg_weight,
                    direction_cfg_scale=direction_cfg_scale,
                    n_steps=base_n_steps,
                    t_end=base_t_end,
                    offload_model=not fast_mode,
                )

                if len(step_scale_list) == 1 and not run_ab_compare and mode == "u_hat":
                    out_file = os.path.join(output_dir, "chord_base_edit.mp4")
                else:
                    out_file = os.path.join(
                        output_dir,
                        f"chord_base_edit_{mode}_step_{cur_step_scale:.2f}.mp4"
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
