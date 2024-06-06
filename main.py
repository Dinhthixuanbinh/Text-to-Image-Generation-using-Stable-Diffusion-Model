
import torch
import os
import torchvision
import numpy as np
from tqdm import tqdm
from config import config
from Noise import LinearNoiseScheduler
from CLIP import get_text_representation, drop_text_condition
from Dataset import CelebDataset
from Unet_model import Unet

from diffusers import VQModel
from transformers import CLIPTokenizer , CLIPTextModel

import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
from PIL import Image




diffusion_config = config['diffusion_params']
dataset_config = config['dataset_params']
diffusion_model_config = config['ldm_params']
train_config = config['train_params']
latents_channel = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the noise scheduler
scheduler = LinearNoiseScheduler(num_timesteps = diffusion_config['num_timesteps'],
                                 beta_start= diffusion_config['beta_start'],
                                 beta_end= diffusion_config['beta_end']
                                 )
# Instantiate Condition related components
text_tokenizer = None
text_model = None
empty_text_embed = None
condition_config = diffusion_model_config['condition_config']
condition_style = condition_config['condition_types']

with torch.no_grad() :
    # Load tokenizer and text model based on config
    text_tokenizer = CLIPTokenizer.from_pretrained('openai /clip-vit-base-patch16')
    text_model = CLIPTextModel.from_pretrained('openai /clip-vit-base-patch16 ').to(device)
    text_model.eval()

    empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)

im_dataset = CelebDataset(split= 'train',
                          im_path= dataset_config['im_path'],
                          im_size= dataset_config['im_size'],
                          im_channels= dataset_config['im_channels'],
                          use_latents= True,
                          latent_path= os.path.join(train_config['task_name'],
                                                    train_config['vqvae_latent_dir_name']),
                                                    condition_config= condition_config
                          )
filters = list(range(0, len(im_dataset), 10))
im_dataset = torch.utils.data.Subset(im_dataset, filters)

data_loader = DataLoader(im_dataset,
                         batch_size= train_config['ldm_batch_size'],
                         shuffle= True)

model = Unet(im_channels= latents_channel,
             model_config= diffusion_model_config
).to(device)

model.train()

num_epochs = train_config['ldm_epochs']
optimizer = Adam(model.parameters(), lr= train_config['ldm_lr'])
criterion = torch.nn.MSELoss()

#Run training
for epoch_idx in range(num_epochs):
    losses = []
    for data in tqdm(data_loader):
        cond_input = None
        if condition_config is not None:
            im, cond_input = data
        else:
            im = data
        optimizer.zero_grad()
        im = im.float().to(device)
    # ########## Handling Conditional Input ###########

        if 'text' in condition_style:
            with torch.no_grad():
                text_condition = get_text_representation(cond_input['text'],
                                                         text_tokenizer,
                                                         text_model,
                                                         device
                )
                text_drop_prob = condition_config['text_condition_config']['cond_drop_prob']
                text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                cond_input['text'] = text_condition
        # ###############################################

        # Sample random noise

        noise = torch.randn_like(im).to(device)

        # Sample timestep

        t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

        # Add noise to images according to timestep

        noisy_im = scheduler.add_noise(im,noise, t)
        noise_pred = model(noisy_im, t, cond_input = cond_input)
        loss = criterion(noise_pred, noise)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    print('Finished epoch :{} | Loss : {:.4f}'.format(
        epoch_idx +1,
        np.mean(losses)
    ))

    torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                train_config['ldm_ckpt_name']

    ))
print('Done Training ....')

# prepare model
model = Unet(im_channels= latents_channel,
             model_config= diffusion_model_config
             ).to(device)

model.eval()
if os.path.exists(os.path.join(train_config['task_name'],
                               train_config['ldm_ckpt_name'])):
    
    print('Loaded unet checkpoint')
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ldm_ckpt_name']
                                                  ), map_location= device))
else :
    raise Exception('Model checkpoint {} not found '.format(os.path.join(train_config['task_name'],
                                                                         train_config['ldm_ckpt_name']
                                                                         )))

vae = VQModel.from_pretrained('CompVis/ldm-celebahq-256', subfolder ='vqvae' )
vae.eval()
vae = vae.to(device)

# genarate
with torch.no_grad():
    xt= torch.randn((1,latents_channel, 64,64)).to(device)
    text_prompt = ['She is a woman with blond hair . She is wearing lipstick.']
    neg_prompt = ['He is a man. ']
    empty_prompt = ['']
    text_prompt_embed = get_text_representation(text_prompt,
                                                text_tokenizer,
                                                text_model,
                                                device)
# Can replace empty prompt with negative prompt

empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
assert empty_text_embed.shape == text_prompt_embed.shape

uncond_input = {
    'text' : empty_text_embed
}
cond_input = {
    'text': text_prompt_embed
}
# ##############################################
# By default classifier free guidance is disabled
# Change value in config or change default value here to enable it
cf_guidance_scale = train_config['cf_guidance_scale']

#################### Sampling Loop ##########################

for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
    # Get prediction of noise
    t = (torch.ones((xt.shape[0],)) * i ).long().to(device)
    noise_pred_cond = model(xt, t, cond_input)

    if cf_guidance_scale > 1:
        noise_pred_uncond = model(xt,t, uncond_input)
        noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond -noise_pred_uncond)
    else:
        noise_pred =  noise_pred_cond
    # Use scheduler to get x0 and xt -1
    xt , x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Save x0
    if i == 0:
        # Decode ONLY the final iamge to save time
        ims = vae.decode(xt).sample
    else:
        ims = x0_pred

    ims = torch.clamp(ims, -1. , 1.).detach().cpu()
    ims = (ims +1) / 2
    grid = make_grid(ims, nrow= 1)
    img = torchvision.transforms.ToPILImage()(grid)

    if not os.path.exists(os.path.join(train_config['task_name'], 'cond_text_samples')):
        os.mkdir(os.path.join(train_config['task_name'], 'cond_text_samples'))
    img.save(os.path.join(train_config['task_name'], 'cond_text_samples', 'x0_{}.png'.format(i)))
    img.close()

# Show image
test_image = Image.open(os.path.join(train_config['task_name'], 'cond_text_samples','x0_{}.png'.format(0) ))