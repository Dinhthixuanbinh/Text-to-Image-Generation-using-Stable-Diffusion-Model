config = {
    " dataset_params ": {
        " im_path ": " data / CelebAMask - HQ " ,
        " im_channels ": 3 ,
        " im_size ": 256 ,
        " name ": " celebhq "
    } ,
    " diffusion_params ": {
        " num_timesteps ": 1000 ,
        " beta_start ": 0.00085 ,
        " beta_end ": 0.012
    } ,
    " ldm_params ": {
        " down_channels ": [128 , 256 , 384 , 512] ,
        " mid_channels ": [512 , 384] ,
        " down_sample ": [ True , True , True ] ,
        " attn_down ": [ True , True , True ] ,
        " time_emb_dim ": 512 ,
        " norm_channels ": 32 ,
        " num_heads ": 16 ,
        " conv_out_channels ": 128 ,
        " num_down_layers ": 2 ,
        " num_mid_layers ": 2 ,
        " num_up_layers ": 2 ,
        " condition_config ": {
            " condition_types ": [" text "] ,
            " text_condition_config ": {
            " text_embed_model ": " clip " ,
            " train_text_embed_model ": False ,
            " text_embed_dim ": 512 ,
            " cond_drop_prob ": 0.1
            }
        }       
    } ,
    " train_params ": {
        " seed ": 1111 ,
        " task_name ": " celebhq " ,
        " ldm_batch_size ": 1 ,
        " ldm_epochs ": 30 ,
        " num_samples ": 1 ,
        " num_grid_rows ": 1 ,
        " ldm_lr ": 0.000005 ,
        " save_latents ": True ,
        " vqvae_latent_dir_name": 'vqvae_latents' ,
        " cf_guidance_scale": 1.0 ,
        " ldm_ckpt_name": " ddpm_ckpt_text_cond_clip . pth " ,
    }
}