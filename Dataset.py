import os
import glob
import pickle
import torchvision
import random
from tqdm import tqdm
from PIL import Image
from torch . utils . data . dataset import Dataset


def load_latents(latent_path):
    '''
Simple utility to save latents to speed up ldm training
: param latent_path :
: return :
    '''
    latents_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k,v in s.items():
            latents_maps[k] = v[0]
    return latents_maps

class CelebDataset(Dataset) :
    '''
    Celeb dataset will by default centre crop and resize the images .
    This can be replaced by any other dataset . As long as all the images
    are under one directory .
    '''
    def __init__ (self, split, im_path, im_size = 256, im_channels = 3, im_ext = 'jpg',
                  use_latents = False, latent_path = None, condition_config = None):
        self.split = split
        if self.split != 'all':
            self.split_filter = pickle.load(open(f'./data/CelebAMask-HQ/{self.split}.pickle ', 'rb'))
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False

        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        self.images , self.texts = self.load_images(im_path)

        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) >= len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latens'. format(len(self.latent_maps)))
            else:
                print('Latents not found')
    def load_images(self, im_path):
        '''
        Gets all images from the path specified
         and stacks them all up
        '''
        assert os.path.exists(im_path), 'images path {} does not exist'.format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path,'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path,'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path,'CelebA-HQ-img/*.{}'.format('jpeg')))

        texts = []
        for fname in tqdm(fnames):
            im_name = os.path.split(fname)[1].split('.')[0]

            if self.split != 'all':
                if im_name  not in self.split_filter:
                    continue
            ims.append(fname)

            if 'text' in self.condition_types:
                captions_im = []
                with open(os.path.join(im_path, 'celeba-caption/{}.txt'.format(im_name))) as f:
                    for line in f.readlines():
                        captions_im.append(line.split())
                texts.append(captions_im)

        if 'text' in self.condition_types:
                assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} captions'.format(len(texts)))

        return ims, texts
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
    # ####### Set Conditioning Info ########

        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = random.sample(self.texts[index], k =1)[0]
        # ######################################
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs    
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input to -1 to 1 range 
            im_tensor = (2 * im_tensor) -1 
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs


