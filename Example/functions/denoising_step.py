"""
Denoising Steps Module for Diffusion-based Fluid Super-resolution

This module implements various sampling strategies for the diffusion model in fluid super-resolution tasks.
It contains both standard diffusion sampling methods (DDIM, DDPM) and physics-guided variants that incorporate
fluid dynamics constraints through gradient information.

Key features:
- Standard deterministic sampling (DDIM)
- Standard stochastic sampling (DDPM)
- Physics-guided deterministic sampling (guided_ddim_steps)
- Physics-guided stochastic sampling (guided_ddpm_steps)

These methods are critical for the reverse diffusion process that generates high-resolution
fluid fields from noise while satisfying physical constraints.
"""
import torch


def compute_alpha(beta, t):
    """
    Compute the cumulative product of (1 - beta) for specified timesteps.
    
    This function calculates alpha_t values which represent the noise schedule
    in the diffusion process. It's a fundamental component for both forward and
    reverse diffusion processes.
    
    Args:
        beta (torch.Tensor): The noise schedule tensor of shape [T].
        t (torch.Tensor): Timestep indices of shape [batch_size].
    
    Returns:
        torch.Tensor: Computed alpha values of shape [batch_size, 1, 1, 1].
        
    Note:
        - Alpha values represent how much of the original signal remains at timestep t
        - We add a zero at the beginning of beta to handle the case when t=0
        - The output shape includes dimensions for spatial components (H, W)
    """
    # Prepend zero to beta sequence for proper indexing
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # Calculate cumulative product of (1-beta) and reshape for broadcasting
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddim_steps(x, seq, model, b, **kwargs):
    """
    Deterministic DDIM sampling process for fluid field generation.
    
    Implements the Denoising Diffusion Implicit Model (DDIM) sampling algorithm,
    which is a deterministic variant of diffusion models. This method enables
    efficient sampling with fewer steps than DDPM while maintaining quality.
    
    Args:
        x (torch.Tensor): Input noise tensor of shape [batch_size, channels, height, width].
        seq (list): Sequence of timesteps for the reverse diffusion process.
        model (nn.Module): The trained diffusion model (U-Net) that predicts noise.
        b (torch.Tensor): Beta schedule defining the noise levels at each timestep.
        **kwargs: Additional arguments:
            - dx_func (callable, optional): Function to compute physical gradients.
            - clamp_func (callable, optional): Function to constrain output values.
            - cache (bool, optional): Whether to store all intermediate steps (True) or only the latest (False).
            - logger (object, optional): Logger object for tracking the sampling process.
            
    Returns:
        tuple: 
            - xs (list): List of generated samples at different denoising steps.
            - x0_preds (list): List of predicted x0 (clean data) at different steps.
            
    Best Practices:
        1. For fluid super-resolution, DDIM is preferred when deterministic results are needed.
        2. Increase the number of sampling steps (via seq) for higher quality reconstruction.
        3. When reconstructing complex turbulent flows, consider using smaller step sizes.
        4. For visualization and analysis, enable cache=True to inspect the entire denoising trajectory.
        5. Use dx_func to incorporate physical constraints where appropriate, especially for 
           higher Reynolds number flows that benefit from physical guidance.
    """
    n = x.size(0)  # Batch size
    seq_next = [-1] + list(seq[:-1])  # Shifted sequence for step pairs
    x0_preds = []  # Store predicted clean images at each step
    xs = [x]  # Store all generated samples (or just the latest if cache=False)
    dx_func = kwargs.get('dx_func', None)  # Function to compute physical gradients
    clamp_func = kwargs.get('clamp_func', None)  # Function to constrain output values
    cache = kwargs.get('cache', False)  # Whether to store all intermediate steps

    # Set up logger if provided
    logger = kwargs.get('logger', None)
    if logger is not None:
        logger.update(x=xs[-1])

    # Iterate through timesteps in reverse order for the denoising process
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            # Current and next timesteps
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            # Compute alpha values for current and next timesteps
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            
            # Current noisy sample
            xt = xs[-1].to('cuda')

            # Predict noise using the diffusion model
            et = model(xt, t)
            
            # Estimate clean data (x0) from noisy data and predicted noise
            # This is the key equation in DDIM: x0 = (xt - sqrt(1-at) * et) / sqrt(at)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            # Coefficient for the noise component in next sample
            c2 = (1 - at_next).sqrt()
            
        # Calculate physical gradient if dx_func is provided
        if dx_func is not None:
            dx = dx_func(xt)  # Get physical correction gradient
        else:
            dx = 0  # No physical correction
            
        with torch.no_grad():
            # Compute next sample using DDIM formula and apply physical gradient correction
            # DDIM formula: x_{t-1} = sqrt(at-1) * x0 + sqrt(1-at-1) * et
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx  # Subtract dx for physical correction
            
            # Apply value constraints if provided
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
                
            # Store the new sample
            xs.append(xt_next.to('cpu'))

        # Update logger if provided
        if logger is not None:
            logger.update(x=xs[-1])

        # If not caching all steps, only keep the latest
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def ddpm_steps(x, seq, model, b,  **kwargs):
    """
    Stochastic DDPM sampling process for fluid field generation.
    
    Implements the original Denoising Diffusion Probabilistic Model (DDPM) sampling algorithm,
    which is stochastic and produces diverse outputs. This is useful when multiple plausible
    reconstructions from the same input are desired.
    
    Args:
        x (torch.Tensor): Input noise tensor of shape [batch_size, channels, height, width].
        seq (list): Sequence of timesteps for the reverse diffusion process.
        model (nn.Module): The trained diffusion model (U-Net) that predicts noise.
        b (torch.Tensor): Beta schedule defining the noise levels at each timestep.
        **kwargs: Additional arguments:
            - dx_func (callable, optional): Function to compute physical gradients.
            - clamp_func (callable, optional): Function to constrain output values.
            - cache (bool, optional): Whether to store all intermediate steps (True) or only the latest (False).
            
    Returns:
        tuple: 
            - xs (list): List of generated samples at different denoising steps.
            - x0_preds (list): List of predicted x0 (clean data) at different steps.
            
    Best Practices:
        1. Use DDPM when diversity in reconstructions is important (e.g., for uncertainty quantification).
        2. For fluid simulations with inherent stochasticity, DDPM can provide more realistic variations.
        3. The stochasticity makes this method suitable for ensemble generation in fluid modeling.
        4. Consider smaller batch sizes compared to DDIM as this method is more memory intensive.
        5. For final evaluations requiring reproducibility, use DDIM instead or fix the random seed.
    """
    n = x.size(0)  # Batch size
    seq_next = [-1] + list(seq[:-1])  # Shifted sequence for step pairs
    xs = [x]  # Store all generated samples (or just the latest if cache=False)
    x0_preds = []  # Store predicted clean images at each step
    betas = b  # Beta schedule for noise levels
    dx_func = kwargs.get('dx_func', None)  # Function to compute physical gradients
    cache = kwargs.get('cache', False)  # Whether to store all intermediate steps
    clamp_func = kwargs.get('clamp_func', None)  # Function to constrain output values

    # Iterate through timesteps in reverse order for the denoising process
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            # Current and next timesteps
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            # Compute alpha values for current and next timesteps
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            
            # Compute beta_t for current step
            beta_t = 1 - at / atm1
            
            # Current noisy sample
            x = xs[-1].to('cuda')

            # Predict noise using the diffusion model
            output = model(x, t.float())
            e = output

            # Estimate clean data (x0) from noisy data and predicted noise
            # This is the key equation in DDPM: x0 = (x/sqrt(at) - sqrt(1/at - 1) * noise)
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)  # Constrain values
            x0_preds.append(x0_from_e.to('cpu'))
            
            # Compute mean for the next step using the posterior q(x_{t-1} | x_t, x_0)
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            
            # Prepare random noise for stochasticity
            noise = torch.randn_like(x)
            
            # Mask to handle the case when t=0 (no noise at the final step)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            
            # Compute log variance according to DDPM formulation
            logvar = beta_t.log()

        # Calculate physical gradient if dx_func is provided
        if dx_func is not None:
            dx = dx_func(x)  # Get physical correction gradient
        else:
            dx = 0  # No physical correction
            
        with torch.no_grad():
            # Compute next sample using DDPM formula and apply physical gradient correction
            # DDPM formula: x_{t-1} = mean + noise * sqrt(variance) - dx
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            
            # Apply value constraints if provided
            if clamp_func is not None:
                sample = clamp_func(sample)
                
            # Store the new sample
            xs.append(sample.to('cpu'))
            
        # If not caching all steps, only keep the latest
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddpm_steps(x, seq, model, b,  **kwargs):
    """
    Physics-guided stochastic DDPM sampling for physically consistent fluid field generation.
    
    This function extends the standard DDPM sampling algorithm by incorporating physical
    constraints through a classifier-free guidance mechanism. It uses model outputs 
    with and without physical gradients, then combines them with a weighting factor.
    
    Args:
        x (torch.Tensor): Input noise tensor of shape [batch_size, channels, height, width].
        seq (list): Sequence of timesteps for the reverse diffusion process.
        model (nn.Module): The trained conditional diffusion model that accepts physical gradients.
        b (torch.Tensor): Beta schedule defining the noise levels at each timestep.
        **kwargs: Additional arguments:
            - dx_func (callable, required): Function to compute physical gradients.
            - w (float, optional): Weighting factor for physical guidance (default: 3.0).
            - clamp_func (callable, optional): Function to constrain output values.
            - cache (bool, optional): Whether to store all intermediate steps.
            
    Returns:
        tuple: 
            - xs (list): List of generated samples at different denoising steps.
            - x0_preds (list): List of predicted x0 (clean data) at different steps.
            
    Raises:
        ValueError: If dx_func is not provided, as physical guidance requires gradient computation.
            
    Best Practices:
        1. Use this method when both physical consistency and solution diversity are required.
        2. The weight parameter 'w' controls the strength of physical guidance:
           - Lower values (0.1-0.5): Subtle guidance, more diversity, possibly less physically accurate
           - Medium values (0.5-2.0): Balanced approach for most applications
           - Higher values (2.0-5.0): Strong physical constraints, less diversity
        3. For high Reynolds number flows, increase 'w' to enforce stronger physical constraints.
        4. Computational cost is higher than regular DDPM as it requires two model evaluations per step.
        5. For ensembles in fluid uncertainty quantification, use moderate 'w' values with multiple runs.
    """
    n = x.size(0)  # Batch size
    seq_next = [-1] + list(seq[:-1])  # Shifted sequence for step pairs
    xs = [x]  # Store all generated samples (or just the latest if cache=False)
    x0_preds = []  # Store predicted clean images at each step
    betas = b  # Beta schedule for noise levels
    dx_func = kwargs.get('dx_func', None)  # Function to compute physical gradients
    if dx_func is None:
        raise ValueError('dx_func is required for guided denoising')
    clamp_func = kwargs.get('clamp_func', None)  # Function to constrain output values
    cache = kwargs.get('cache', False)  # Whether to store all intermediate steps
    w = kwargs.get('w', 3.0)  # Physics guidance weight

    # Iterate through timesteps in reverse order for the denoising process
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            # Current and next timesteps
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            # Compute alpha values for current and next timesteps
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            
            # Compute beta_t for current step
            beta_t = 1 - at / atm1
            
            # Current noisy sample
            x = xs[-1].to('cuda')

        # Calculate physical gradient using the provided function
        dx = dx_func(x)  # Get physical correction gradient
        
        with torch.no_grad():
            # Apply classifier-free guidance with physical information
            # Formula: (w+1)*model(x,t,dx) - w*model(x,t)
            # This emphasizes the direction suggested by physical gradients
            output = (w+1)*model(x, t.float(), dx)-w*model(x, t.float())
            e = output

            # Estimate clean data (x0) from noisy data and guided noise prediction
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)  # Constrain values
            x0_preds.append(x0_from_e.to('cpu'))
            
            # Compute mean for the next step using the posterior q(x_{t-1} | x_t, x_0)
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            
            # Prepare random noise for stochasticity
            noise = torch.randn_like(x)
            
            # Mask to handle the case when t=0 (no noise at the final step)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            
            # Compute log variance according to DDPM formulation
            logvar = beta_t.log()

        with torch.no_grad():
            # Compute next sample using guided DDPM formula and apply physical gradient correction
            # Formula: mean + noise * sqrt(variance) - dx
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            
            # Apply value constraints if provided
            if clamp_func is not None:
                sample = clamp_func(sample)
                
            # Store the new sample
            xs.append(sample.to('cpu'))
            
        # If not caching all steps, only keep the latest
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddim_steps(x, seq, model, b, **kwargs):
    """
    Physics-guided deterministic DDIM sampling for physically consistent fluid field generation.
    
    This function is the cornerstone of PhyDiff model as described in the paper. It enhances
    the standard DDIM sampling with physical guidance through classifier-free guidance and
    direct physical gradient correction. This produces deterministic, physically accurate
    fluid reconstructions.
    
    Args:
        x (torch.Tensor): Input noise tensor of shape [batch_size, channels, height, width].
        seq (list): Sequence of timesteps for the reverse diffusion process.
        model (nn.Module): The trained conditional diffusion model that accepts physical gradients.
        b (torch.Tensor): Beta schedule defining the noise levels at each timestep.
        **kwargs: Additional arguments:
            - dx_func (callable, required): Function to compute physical gradients (typically vorticity_residual).
            - w (float, optional): Weighting factor for physical guidance (default: 3.0).
            - clamp_func (callable, optional): Function to constrain output values.
            - cache (bool, optional): Whether to store all intermediate steps.
            - logger (object, optional): Logger object for tracking the sampling process.
            
    Returns:
        tuple: 
            - xs (list): List of generated samples at different denoising steps.
            - x0_preds (list): List of predicted x0 (clean data) at different steps.
            
    Raises:
        ValueError: If dx_func is not provided, as physical guidance requires gradient computation.
            
    Best Practices:
        1. This is the recommended method for high-fidelity fluid super-resolution reconstruction.
        2. Tune the guidance weight 'w' based on Reynolds number and complexity:
           - For Re < 500: w = 0.5-1.5 typically works well
           - For Re > 1000: w = 2.0-4.0 may be needed to enforce physical consistency
           - For complex geometries: increase w gradually until physical constraints are satisfied
        3. Optimize the number of sampling steps (via seq) for your specific application:
           - 50-100 steps for quick preliminary results
           - 100-200 steps for publication-quality reconstructions
        4. For scientific analysis, enable cache=True to visualize the entire denoising trajectory.
        5. This method combines well with multi-scale or progressive refinement approaches.
    """
    n = x.size(0)  # Batch size
    seq_next = [-1] + list(seq[:-1])  # Shifted sequence for step pairs
    x0_preds = []  # Store predicted clean images at each step
    xs = [x]  # Store all generated samples (or just the latest if cache=False)
    dx_func = kwargs.get('dx_func', None)  # Function to compute physical gradients
    if dx_func is None:
        raise ValueError('dx_func is required for guided denoising')
    clamp_func = kwargs.get('clamp_func', None)  # Function to constrain output values
    cache = kwargs.get('cache', False)  # Whether to store all intermediate steps
    w = kwargs.get('w', 3.0)  # Physics guidance weight
    
    # Set up logger if provided
    logger = kwargs.get('logger', None)
    if logger is not None:
        logger.update(x=xs[-1])

    # Iterate through timesteps in reverse order for the denoising process
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            # Current and next timesteps
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            
            # Compute alpha values for current and next timesteps
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            
            # Current noisy sample
            xt = xs[-1].to('cuda')

        # Calculate physical gradient using the provided function (typically vorticity_residual)
        dx = dx_func(xt)  # Get physical correction gradient from fluid dynamics PDEs

        with torch.no_grad():
            # Apply classifier-free guidance with physical information
            # Formula: (w+1)*model(xt,t,dx) - w*model(xt,t)
            # This emphasizes the predictions that align with physical constraints
            et = (w+1)*model(xt, t, dx) - w*model(xt, t)

            # Estimate clean data (x0) from noisy data and guided noise prediction
            # This is the key equation in DDIM: x0 = (xt - sqrt(1-at) * et) / sqrt(at)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            # Coefficient for the noise component in next sample
            c2 = (1 - at_next).sqrt()

        with torch.no_grad():
            # Compute next sample using guided DDIM formula and apply physical gradient correction
            # Formula: sqrt(at_next) * x0_t + sqrt(1-at_next) * et - dx
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            
            # Apply value constraints if provided
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
                
            # Store the new sample
            xs.append(xt_next.to('cpu'))

        # Update logger if provided
        if logger is not None:
            logger.update(x=xs[-1])

        # If not caching all steps, only keep the latest
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds