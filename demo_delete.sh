export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=2 --master_port 11245 generate.py \
         --task i2v-14B \
         --size 832*480 \
         --ckpt_dir ../Wan2.1-I2V-14B-480P \
         --image_origin "./examples/delete/07/07.png" \
         --image "./examples/delete/07/07_delete.png" \
         --pnp --t5_cpu --dit_fsdp --ulysses_size 2 \
         --pnp_layers 21 26 27 29\
         --injection_step 0.5 \
         --prompt_origin "" \
         --prompt "A mature man with silver hair and dark sunglasses, dressed in a light grey suit, white shirt, and cream sweater, confidently walks along a dirt path in a park, holding a black umbrella and a red and white cane." \
         --sample_solver "fm_new" \
         --sample_steps 51 \
         --offload_model 'True' \
         --load_intermediate_latent_path "../Wan2.1_delete/latents_delete/07.pt"\
         --load_intermediate_latent_t 995.9473266601562 \
         --sample_guide_scale 3.0 \
         --is_delete

torchrun --nproc_per_node=2 --master_port 11245 generate.py \
         --task i2v-14B \
         --size 832*480 \
         --ckpt_dir ../Wan2.1-I2V-14B-480P \
         --image_origin "./examples/delete/19/19.png" \
         --image "./examples/delete/19/19_delete.png" \
         --pnp --t5_cpu --dit_fsdp --ulysses_size 2 \
         --pnp_layers 21 26 27 29\
         --injection_step 0.5 \
         --prompt_origin "" \
         --prompt "A cow is standing in the fence grazing. There is straw on the ground and a trough." \
         --sample_solver "fm_new" \
         --sample_steps 51 \
         --offload_model 'True' \
         --load_intermediate_latent_path "../Wan2.1_delete/latents_delete/19.pt"\
         --load_intermediate_latent_t 995.9473266601562 \
         --sample_guide_scale 3.0 \
         --is_delete