import torch
import os
import glob
import pickle
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from diffusers import VQModel

from config import config
from Dataset import CelebDataset

dataset_config = config['dataset_params']
train_config = config['train_params']
num_images = train_config['num_samples']
ngrid = train_config['num_grid_rows']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
im_dataset = CelebDataset(split= 'all',
                      im_path= dataset_config['im_path'],
                      im_size= dataset_config['im_size'],
                      im_channels= dataset_config['im_channels'])

data_loader = DataLoader(im_dataset, batch_size= 2, shuffle= True)

idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
ims = torch.cat([im_dataset[idx][None , :] for idx in idxs])
ims = ims.to(device)

vae = VQModel.from_pretrained('CompVis/ldm-celebahq-256', subforder = "vqvae")
vae.eval()
vae = vae.to(device)

os.makedirs(os.path.join(train_config['task_name']), exist_ok= True)
with torch.no_grad():
    encoded_output = vae.encode(ims).latents
    decoded_output = vae.decode(encoded_output).sample
    encoded_output = torch.clamp(encoded_output, -1. , 1.)
    encoded_output = (encoded_output + 1) / 2
    decoded_output = torch.clamp(encoded_output, -1. , 1.)
    decoded_output = (decoded_output + 1) / 2
    ims = (ims + 1) / 2

    encoder_grid = make_grid(encoded_output.cpu(), nrow= ngrid)
    decoder_grid = make_grid(decoded_output.cpu(), nrow= ngrid)
    input_grid = make_grid(ims.cpu(), nrow= ngrid)
    encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
    decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
    input_grid = torchvision.transforms.ToPILImage()(input_grid)

    input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
    encoder_grid.save(os.path.join(train_config['task_name'], 'encoder_samples.png'))
    decoder_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))

    os.makedirs(os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name']), exist_ok= True)
    if train_config['save_latents'] : 
        # save Latents (but in a very unoptimized way)
        latent_path = os.path.join(train_config['task_name'],
                                   train_config['vqvae_latent_dir_name'])
        latent_fnames = glob.glob(os.path.join(train_config['task_name'], 
                                               train_config['vqvae_latent_dir_name'], '*.pkl'))
        assert len(latent_fnames) == 0, ' Latents already present . Delete all latent files and re -run'
        if not os.path.exists(latent_path):
            os.mkdir(latent_path)
        print('Saving Latents for {}'.format(dataset_config['name']))

        fname_latent_map = {}
        part_count = 0
        count = 0 
        for idx, im in enumerate(tqdm(data_loader)):
            encoded_output = vae.encode(im.float().to(device)).latents
            fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
            # Save latents every 1000 images
            if (count + 1) % 1000 == 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path, 
                                                                '{}.pkl'.format(part_count)), 'wb'))
                part_count += 1
                fname_latent_map = {}
            count += 1

        if len(fname_latent_map) > 0 :
            pickle.dump(fname_latent_map, open(os.path.join(latent_path, 
                                                                '{}.pkl'.format(part_count)), 'wb'))
        print('Done saving latents')