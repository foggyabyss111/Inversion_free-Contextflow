export CUDA_VISIBLE_DEVICES="0,1"

# For insertion or swapping, set frame_num = 81; for deletion, set frame_num = 49
torchrun --nproc_per_node=2 generate.py \
         --task i2v-14B \
         --size 480*832 \
         --ckpt_dir /data1/vision/Wan2.1-I2V-14B-480P \
         --image ./examples/insert/59852/59852.jpg \
         --video ./examples/insert/59852/59852.mp4 \
         --latent_name latents_insert/59852.pt \
         --reconstruction --frame_num 81 \
         --t5_cpu \
         --dit_fsdp \
         --ulysses_size 2 \
         --prompt_origin "" \
         --prompt "" \
         --sample_solver "fm_new" \
         --sample_steps 51 \
         --offload_model 'True'