import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    parser.add_argument('--im_path', type=str, default='data/CelebAMask-HQ')
    parser.add_argument('--im_channels', type=int, default=3)
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--name', type=str, default='celebhq')
    
    # Diffusion parameters
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.00085)
    parser.add_argument('--beta_end', type=float, default=0.012)
    
    # LDM parameters
    parser.add_argument('--down_channels', type=int, nargs='+', default=[256, 384, 512, 768])
    parser.add_argument('--mid_channels', type=int, nargs='+', default=[768, 512])
    parser.add_argument('--down_sample', type=bool, nargs='+', default=[True, True, True])
    parser.add_argument('--attn_down', type=bool, nargs='+', default=[True, True, True])
    parser.add_argument('--time_emb_dim', type=int, default=512)
    parser.add_argument('--norm_channels', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--conv_out_channels', type=int, default=128)
    parser.add_argument('--num_down_layers', type=int, default=2)
    parser.add_argument('--num_mid_layers', type=int, default=2)
    parser.add_argument('--num_up_layers', type=int, default=2)
    
    # Condition configuration
    parser.add_argument('--condition_types', type=str, nargs='+', default=['text'])
    parser.add_argument('--text_embed_model', type=str, default='clip')
    parser.add_argument('--train_text_embed_model', type=bool, default=False)
    parser.add_argument('--text_embed_dim', type=int, default=512)
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--task_name', type=str, default='celebhq')
    parser.add_argument('--ldm_batch_size', type=int, default=16)
    parser.add_argument('--ldm_epochs', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_grid_rows', type=int, default=1)
    parser.add_argument('--ldm_lr', type=float, default=0.000005)
    parser.add_argument('--save_latents', type=bool, default=True)
    parser.add_argument('--vqvae_latent_dir_name', type=str, default='vqvae_latents')
    parser.add_argument('--cf_guidance_scale', type=float, default=1.0)
    parser.add_argument('--ldm_ckpt_name', type=str, default='ddpm_ckpt_text_cond_clip.pth')
    
    args = parser.parse_args()
    return args

