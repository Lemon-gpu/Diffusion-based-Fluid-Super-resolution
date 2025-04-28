import torch  # Import PyTorch library for deep learning operations

"""
denoising.py - Core implementation of the diffusion model sampling process
This module implements the reverse process from noise to clean image in diffusion models, including:
1. Computing noise scale parameter alpha in the diffusion process
2. Implementing the DDIM sampling algorithm (generalized_steps)
3. Implementing the original DDPM sampling algorithm (ddpm_steps)
These functions play a key role in training and inference for fluid super-resolution tasks
"""

def compute_alpha(beta, t):
    """
    Computes the alpha parameter in the diffusion process, used to adjust noise levels
    
    This is a key parameter in diffusion models, representing the proportion of original signal
    preserved at time step t. Alpha is calculated as the cumulative product from the beta parameter
    (noise variance added at each step).
    
    Algorithm steps:
    1. Prepend a zero to the beta sequence to ensure index alignment
    2. Calculate the cumulative product of (1-beta), yielding alpha values that decrease over time
    3. Select the alpha value corresponding to the input time step t
    4. Reshape for subsequent calculations
    
    Parameters:
        beta (torch.Tensor): Noise variance parameters for each step in the diffusion process
        t (torch.Tensor): The specified time step
        
    Returns:
        torch.Tensor: Alpha value with shape (-1,1,1,1), corresponding to noise level at time t
    """
    # Prepend a zero to beta for indexing alignment (beta[0] = 0)
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    
    # Calculate cumulative product of (1-beta), select values at t+1, and reshape to (-1,1,1,1)
    # This computes α_t = ∏_{s=1}^{t} (1-β_s) which represents signal retention at time t
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, b, **kwargs):
    """
    Implements the DDIM (Denoising Diffusion Implicit Model) sampling algorithm, providing
    controllable balance between deterministic and stochastic behavior
    
    This is an improved sampling method that allows controlling randomness during sampling
    and can skip intermediate steps to improve sampling efficiency. When eta=0, the sampling path
    is completely deterministic; when eta=1, it's equivalent to DDPM stochastic sampling.
    
    Algorithm flow:
    1. Start from noise and reverse through the time sequence
    2. At each time step:
       - Use the model to predict noise
       - Estimate the original noiseless image
       - Calculate the state for the next time step
       - Apply controllable randomness
    
    Parameters:
        x (torch.Tensor): Initial noise, typically a standard normal distribution sample
        seq (list): Sampling time step sequence, usually in descending order
        model (nn.Module): Trained denoising network model
        b (torch.Tensor): Beta parameter sequence controlling noise variance
        **kwargs: Optional parameters, including eta (coefficient controlling randomness)
        
    Returns:
        tuple: (sampling sequence, noiseless image prediction at each step)
    """
    with torch.no_grad():  # Disable gradient computation for inference
        n = x.size(0)  # Get batch size from input tensor
        
        # Create sequence for the next timestep, starting with -1 (end state)
        seq_next = [-1] + list(seq[:-1])
        
        x0_preds = []  # List to store predictions of x0 (clean image) at each step
        xs = [x]  # List to store noisy samples at each step, initialized with input noise
        
        # Iterate through time steps in reverse order (from noisy to clean)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # Create tensors for current and next timestep
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            # Calculate alpha values for current and next timestep
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            
            # Get current noisy sample and move to GPU
            xt = xs[-1].to('cuda')
            
            # Predict noise (et) using the model at current timestep
            et = model(xt, t)
            
            # Estimate the clean image x0 from the current noisy sample xt and predicted noise et
            # This uses the reparameterization formula: x0 = (xt - sqrt(1-αt) * et) / sqrt(αt)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))  # Store clean image prediction on CPU
            
            # Calculate coefficient c1 based on eta parameter (controls sampling stochasticity)
            # When eta=0, c1=0 and sampling becomes deterministic (DDIM)
            # When eta=1, sampling is equivalent to DDPM
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            
            # Calculate coefficient c2 to ensure variance properly scales
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            # Generate next noisy sample using DDIM update formula:
            # x_{t-1} = sqrt(α_{t-1}) * x0 + c1 * random_noise + c2 * predicted_noise
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))  # Store sample on CPU to save GPU memory

    return xs, x0_preds  # Return all samples and clean image predictions


def ddpm_steps(x, seq, model, b, **kwargs):
    """
    Implements the original DDPM (Denoising Diffusion Probabilistic Models) sampling algorithm
    
    This is the standard sampling method for diffusion models, following fixed stochasticity
    and gradually removing noise in a Markov chain manner. Unlike generalized_steps, ddpm_steps
    introduces a fixed degree of randomness at each step that cannot be adjusted.
    
    Algorithm flow:
    1. Start from Gaussian noise and reverse through the time sequence
    2. At each time step:
       - Use the model to predict current noise
       - Estimate the original noiseless image
       - Calculate the mean for the next time step according to the formula
       - Add random Gaussian noise, with magnitude controlled by beta parameters
    
    Parameters:
        x (torch.Tensor): Initial noise tensor
        seq (list): Sampling time step sequence in descending order
        model (nn.Module): Trained denoising network model
        b (torch.Tensor): Beta parameter sequence
        **kwargs: Additional parameters (unused in this function)
        
    Returns:
        tuple: (sampling sequence, noiseless image prediction at each step)
    """
    with torch.no_grad():  # Disable gradient computation for inference
        n = x.size(0)  # Get batch size from input tensor
        
        # Create sequence for the next timestep, starting with -1 (end state)
        seq_next = [-1] + list(seq[:-1])
        
        xs = [x]  # List to store noisy samples at each step, initialized with input noise
        x0_preds = []  # List to store predictions of x0 (clean image) at each step
        betas = b  # Beta schedule for noise levels
        
        # Iterate through time steps in reverse order (from noisy to clean)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # Create tensors for current and next timestep
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            # Calculate alpha values for current and next timestep
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            
            # Calculate beta_t for current step (variance for noise addition)
            beta_t = 1 - at / atm1
            
            # Get current noisy sample and move to GPU
            x = xs[-1].to('cuda')

            # Predict noise using the model at current timestep
            output = model(x, t.float())
            e = output

            # Estimate clean image x0 from noisy sample x and predicted noise e
            # Using the reparameterization formula: x0 = (x - sqrt(1-αt) * e) / sqrt(αt)
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            
            # Clamp the estimated x0 to valid image range [-1, 1]
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))  # Store clean image prediction on CPU
            
            # Calculate mean for the next timestep using a weighted combination of:
            # 1. The predicted clean image x0_from_e
            # 2. The current noisy sample x
            # This formula comes from the DDPM paper's posterior q(x_{t-1}|x_t,x_0)
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            # Set the mean for sampling the next timestep
            mean = mean_eps
            
            # Generate random noise for adding stochasticity
            noise = torch.randn_like(x)
            
            # Create mask to prevent noise addition at t=0 (final step)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            
            # Calculate log variance for the noise scaling
            logvar = beta_t.log()
            
            # Generate the next sample using the mean and scaled noise
            # x_{t-1} = mean + mask * exp(0.5 * logvar) * noise
            # The mask ensures no noise is added at the final step
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))  # Store sample on CPU to save GPU memory
            
    return xs, x0_preds  # Return all samples and clean image predictions
