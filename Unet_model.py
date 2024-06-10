import torch
import torch.nn as nn
from blocks import DownBlock , MidBlock , UpBlockUnet
from blocks import get_time_embedding

class Unet(nn.Module):
    '''
    Unet model comprising Down blocks , Midblocks and Uplocks
    '''
    def __init__(self, im_channels, model_config):
        super().__init__
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.dowm_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']

        #  Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.dowm_sample) == len(self.down_channels) -1
        assert len(self.attns) == len(self.down_channels) - 1

        ###### Class , Mask and Text Conditioning Config #####
        self.condition_config = model_config['condition_config']
        self.text_cond = True
        self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
        self.cond = self.text_cond
        # ##################################

        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0],kernel_size= 3, padding= 1)
        # Initial projection from sinusoidal time embedding

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        self.up_sample = list(reversed(self.dowm_sample))
        self.downs = nn.ModuleList([])

        # Build the Downblocks
        for i in range(len(self.down_channels) - 1):
            # Cross Attention and Context Dim only needed if text condition is present
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i +1], self.t_emb_dim,
                                        down_sample = self.dowm_sample[i],
                                        num_heads = self.num_heads,
                                        num_layers = self.num_down_layers,
                                        attn = self.attns[i] , 
                                        norm_channels = self.norm_channels,
                                        cross_attn = self.cond,
                                        context_dim = self.text_embed_dim
                                         ))
            self.mids = nn.ModuleList([])
            # Build the Midblocks
            for i in range(len(self.mid_channels) - 1):
                self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i +1], self.t_emb_dim,
                                        num_heads = self.num_heads,
                                        num_layers = self.num_mid_layers,
                                        norm_channels = self.norm_channels,
                                        cross_attn = self.text_cond,
                                        context_dim = self.text_embed_dim))
            self.ups = nn.ModuleList([])
            #  build the Upblocks

            for i in reversed(range(len(self.down_channels)- 1)):
                self.ups.append(
                    UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i -1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample = self.dowm_sample[i],
                            num_heads = self.num_heads,
                            num_layers = self.num_up_layers,
                            norm_channels = self.norm_channels,
                            cross_attn = self.text_cond,
                            context_dim = self.text_embed_dim
                                ))
                
            self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
            self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond_input = None):
        # Shapes assuming downblocks are [C1 , C2 , C3 , C4]
        # Shapes assuming midblocks are [C4 , C4 , C3]
        # B x C x H x W -> # B x C1 x H x W
        out = self.conv_in(x)

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tesor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        context_hidden_states = None
        context_hidden_states = cond_input['text']
        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)
        # down_outs [B x C1 x H x W, B x C2 x H/2 x W/2 , B x C3 x H/4 x W /4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
        #    out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
        #  out [B x C2 x H/4 x W/4 , B x C1 x H/2 x W/2 , B x 16 x H x W]

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out


