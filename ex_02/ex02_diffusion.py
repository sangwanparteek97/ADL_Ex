import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    t = torch.arange(timesteps)
    factor = ((t / timesteps)+s)/(1+s)
    alpha = torch.cos(factor * (torch.pi / 2)) ** 2
    beta_schedule = 1 - torch.divide(alpha[:-1], alpha[1:])
    #beta_schedule = torch.cat([torch.zeros(1), beta_schedule])  # Set initial beta to 0
    beta_schedule = torch.clamp(beta_schedule, max=0.999)  # Clipping beta to prevent singularities at the end
    return beta_schedule



def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    slimit = 6
    t = torch.arange(timesteps)
    input = -slimit + (2 * t / timesteps * slimit)
    sigmoid_output = torch.sigmoid(input)
    beta_schedule = beta_start + sigmoid_output * (beta_end - beta_start)
    return beta_schedule


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda",classifier_free_guidence =False):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.classifier_free_guidence = classifier_free_guidence
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help
        self.central_betas = torch.sqrt(torch.cumprod(1-self.betas, dim=0))

        # define alphas
        # TODO
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        # self.sqrt_alphas = torch.sqrt(self.betas)
        # self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.betas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.sqrt_reciprocal_betas = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index,c=None,w=0.5):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension
        # Equation 11 in the paper
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        xt = x
        theta_xt = model(xt, t,c)
        if self.classifier_free_guidence:
            beta_xt = model(xt,t,c)
            xt_new = (1+w)* beta_xt -w * theta_xt
        else:
            xt_new = theta_xt
        # Use our model (noise predictor) to predict the mean
        x_t_1 = sqrt_recip_alphas_t * (xt - betas_t * xt_new / sqrt_one_minus_alphas_cumprod_t)
        # x_t_1 = (1.0 / self.sqrt_alphas_cumprod[t_index]) * (
        #             xt - (self.betas[t_index] / self.sqrt_one_minus_alphas_cumprod[t_index]) * xt_new)
        # TODO (2.2): The method should return the image at timestep t-1.
        if t_index == 0:
            return x_t_1
        else:
            sqrt_reciprocal_betas = extract(self.sqrt_reciprocal_betas, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return x_t_1 + torch.sqrt(sqrt_reciprocal_betas) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        # noise = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        # z = torch.zeros_like(noise)  ##zeros or rand
        # Images = []
        # # TODO (2.2): Return the generated images
        # for t in reversed(range(self.timesteps)):
        #     xt = noise
        #     if t > 0:
        #         z = torch.randn_like(noise)
        #
        #     theta_xt = model(xt, t)
        #     x_t_1 = torch.sqrt(1.0 / self.alphas[t]) * (
        #                 xt - ((1.0 - self.alphas[t]) / self.central_betas[t] ) * theta_xt) + \
        #             self.sqrt_reciprocal_betas[t] * z
        #     noise = x_t_1
        #     Images.append(x_t_1)
        #
        # return Images
        device = next(model.parameters()).device
        b = batch_size
        img = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)
        t = torch.tensor(t)
        # sqrt_alphas_cumprod_t = torch.sqrt(torch.cumprod(self.alphas[:t], dim=0))
        # sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_zero.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape)



        x_t =  sqrt_alphas_cumprod_t * x_zero + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1",classes=None): #self, x_start, t, *, classes, noise = None
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            x_noisy = self.q_sample(x_zero, t)
        else:
            x_noisy = self.q_sample(x_zero=x_zero, t=t, noise=noise)


        if self.classifier_free_guidence:
            noise_pred = denoise_model(x_noisy, t,classes = classes)
        else:
            noise_pred = denoise_model(x_noisy, t,classes)
        #print(f"noise predicted shape: {noise_pred.shape}, target noise : {noise}")
        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(x_noisy, noise_pred)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(x_noisy, noise_pred)
        else:
            raise NotImplementedError()

        return loss
