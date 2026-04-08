export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=2 --master_port 11245 generate.py \
         --task i2v-14B \
         --size 832*480 \
         --ckpt_dir ../Wan2.1-I2V-14B-480P \
         --image_origin "./examples/insert/407483/407483.jpg" \
         --image "./examples/insert/407483/407483_monkey2.jpg" \
         --pnp --t5_cpu --dit_fsdp --ulysses_size 2 \
         --pnp_layers 6 7 8 9 \
         --injection_step 0.5 \
         --prompt_origin "" \
         --prompt "A red poppy flower surrounded by purple flowers. A large gorilla is gently trying to touch the red poppy flower." \
         --sample_solver "fm_new" \
         --sample_steps 51 \
         --offload_model 'True' \
         --load_intermediate_latent_path "../latents/407483.pt"\
         --load_intermediate_latent_t 995.9473266601562 \
         --sample_guide_scale 3.0

torchrun --nproc_per_node=2 --master_port 11245 generate.py \
         --task i2v-14B \
         --size 832*480 \
         --ckpt_dir ../Wan2.1-I2V-14B-480P \
         --image_origin "./examples/insert/520004/520004.jpg" \
         --image "./examples/insert/520004/520004_car.jpg" \
         --pnp --t5_cpu --dit_fsdp --ulysses_size 2 \
         --pnp_layers 6 7 8 9\
         --injection_step 0.5 \
         --prompt_origin "" \
         --prompt "A blue car is driving on the road." \
         --sample_solver "fm_new" \
         --sample_steps 51 \
         --offload_model 'True' \
         --load_intermediate_latent_path "../latents/520004.pt"\
         --load_intermediate_latent_t 995.9473266601562 \
         --sample_guide_scale 3.0
