import torch 
class LinearNoiseScheduler:
    '''
    Linear noise scheduler c s d ng trong DDPM
    '''
    def __init__(self, num_timesteps,  beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = (
            torch.linspace(beta_end ** 0.5, beta_end ** 0.5 , num_timesteps) ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim= 0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]   
        sqrt_alpha_cum_prod  = self.sqrt_alpha_cum_prod(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod(original.device)[t].reshape(batch_size)

    #  Reshape till (B ,) becomes (B ,1 ,1 ,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        return (sqrt_alpha_cum_prod.to(original.device) * original + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    def sample_prev_timestep(self , xt , noise_pred , t):
        '''
        Use the noise prediction by model to get
         xt -1 using xt and the nosie predicted
         : param xt: current timestep sample
         : param noise_pred : model noise prediction
         : param t: current timestep we are at
         : return :
        '''
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)
               )/
               torch.sqrt(self.sqrt_alpha_cum_prod.to(xt.device)[t])
               )
        x0 = torch.clamp(x0, -1.0, 1.0)
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean , x0
        else:
            variance = (1- self.alpha_cum_prod.to(xt.device)[t-1] ) /(1- self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.rand(xt.shape).to(xt.device)

        return mean + sigma * z , x0