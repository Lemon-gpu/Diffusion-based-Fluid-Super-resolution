"""
Physics-Informed Diffusion Model Loss Functions Module

This module implements the core loss functions for the physics-informed diffusion model
for fluid flow super-resolution. It contains:

1. A physics-based residual calculation function (voriticity_residual) that enforces
   the 2D vorticity equation constraints
2. Standard noise estimation loss for diffusion models
3. Conditional noise estimation loss that combines data-driven learning with physics constraints

The physics guidance is implemented through the vorticity equation in its residual form,
which is a key innovation in this framework compared to purely data-driven approaches.
"""

import torch
import numpy as np

def voriticity_residual(w, re=1000.0, dt=1/32, calc_grad=True):
    """
    Calculate the residual of the 2D vorticity equation and its gradient.
    
    This function computes the residual of the vorticity transport equation:
    ∂w/∂t + u·∇w = (1/Re)∇²w - βw + f
    
    where:
    - w is the vorticity field
    - u = (u,v) is the velocity field derived from the stream function
    - Re is the Reynolds number
    - β is a damping coefficient (set to 0.1)
    - f is the external forcing term
    
    The function uses spectral methods (FFT) for efficient and accurate computation
    of spatial derivatives.
    
    Args:
        w (torch.Tensor): Vorticity field tensor of shape [batch_size, time_steps, height, width]
        re (float): Reynolds number, controlling the viscosity term. Default: 1000.0
        dt (float): Time step size for temporal derivative calculation. Default: 1/32
        calc_grad (bool): Whether to calculate gradient of residual. Default: True
        
    Returns:
        torch.Tensor: Gradient of the residual with respect to input vorticity field
                     Shape matches input tensor w
    
    Implementation Details:
        - Stream function is calculated by solving the Poisson equation ∇²ψ = -w
        - Velocity components are derived from the stream function as u = ∂ψ/∂y, v = -∂ψ/∂x
        - All spatial derivatives are computed in Fourier space for accuracy
        - Temporal derivatives use central difference scheme
    """
    # Extract batch size from input vorticity tensor
    batchsize = w.size(0)
    
    # Clone input tensor and enable gradient tracking for backpropagation
    w = w.clone()
    w.requires_grad_(True)
    
    # Extract spatial dimensions of the vorticity field
    nx = w.size(2)  # Number of grid points in x-direction
    ny = w.size(3)  # Number of grid points in y-direction
    device = w.device  # Get device (CPU/GPU) of input tensor
    
    # Transform vorticity field to Fourier space (spectral domain)
    # Note: we use interior time steps (1:-1) to calculate time derivatives later
    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    
    # Set up wavenumbers for spectral differentiation
    # These represent frequencies in Fourier space
    k_max = nx//2  # Maximum wavenumber (Nyquist frequency)
    N = nx  # Grid size
    
    # Create wavenumber mesh in x-direction: [0, 1, ..., k_max-1, -k_max, ..., -1]
    # Reshape to match the 4D tensor format [batch, channel, height, width]
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    
    # Create wavenumber mesh in y-direction with same pattern but transposed
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    
    # Create Laplacian operator in Fourier space: -(k_x² + k_y²)
    # This is the spectral representation of the ∇² operator
    lap = (k_x ** 2 + k_y ** 2)
    
    # Avoid division by zero at DC component (0,0)
    # This effectively sets the mean of the stream function to zero
    lap[..., 0, 0] = 1.0
    
    # Solve Poisson equation ∇²ψ = -w for stream function ψ
    # In Fourier space, this is a simple division: ψ_h = w_h/(-∇²)
    psi_h = w_h / lap

    # Calculate velocity components from stream function
    # u = ∂ψ/∂y, v = -∂ψ/∂x (in Fourier space: multiplication by wavenumbers)
    # The 1j factor represents the imaginary unit for derivative calculation
    u_h = 1j * k_y * psi_h  # x-component of velocity in Fourier space
    v_h = -1j * k_x * psi_h  # y-component of velocity in Fourier space
    
    # Calculate spatial derivatives of vorticity
    # wx = ∂w/∂x, wy = ∂w/∂y (in Fourier space)
    wx_h = 1j * k_x * w_h  # x-derivative of vorticity in Fourier space
    wy_h = 1j * k_y * w_h  # y-derivative of vorticity in Fourier space
    
    # Calculate Laplacian of vorticity (∇²w) for viscous term
    # In Fourier space, Laplacian is multiplication by -(k_x² + k_y²)
    wlap_h = -lap * w_h

    # Transform the computed quantities back to physical space using inverse FFT
    # Note: We only need the real part, and only up to k_max+1 frequencies due to symmetry
    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])  # x-velocity component
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])  # y-velocity component
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])  # x-derivative of vorticity
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])  # y-derivative of vorticity
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])  # Laplacian of vorticity
    
    # Calculate advection term (u·∇w = u*wx + v*wy) - nonlinear transport of vorticity
    advection = u*wx + v*wy

    # Calculate time derivative using central difference scheme
    # ∂w/∂t ≈ (w(t+dt) - w(t-dt))/(2*dt)
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # Set up forcing term f = -4*cos(4y) - external force driving the flow
    # Create a spatial grid spanning [0, 2π] in both directions
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]  # Remove last point (same as first due to periodicity)
    X, Y = torch.meshgrid(x, x)  # Create 2D coordinate grid
    f = -4*torch.cos(4*Y)  # Forcing term varying in y-direction

    # Calculate the residual of the vorticity equation
    # residual = ∂w/∂t + u·∇w - (1/Re)∇²w + βw - f
    # Each term corresponds to a physical process:
    # - wt: temporal evolution
    # - advection: nonlinear transport by velocity field
    # - (1.0/re)*wlap: viscous diffusion
    # - 0.1*w: linear damping (β=0.1)
    # - f: external forcing
    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    
    # Calculate mean squared error of the residual for loss function
    residual_loss = (residual**2).mean()
    
    # Calculate gradient of the residual loss with respect to input vorticity field
    # This gradient will be used to guide the diffusion model toward physically valid solutions
    dw = torch.autograd.grad(residual_loss, w)[0]
    return dw

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    """
    Standard diffusion model noise estimation loss function.
    
    This function implements the basic denoising score matching loss for diffusion models,
    where the model learns to predict the noise that was added to the data.
    
    The loss is calculated as the mean squared error between the predicted noise
    and the actual noise used in the forward diffusion process.
    
    Args:
        model (nn.Module): The denoising network (U-Net) that predicts noise
        x0 (torch.Tensor): Clean data samples of shape [batch_size, channels, height, width]
        t (torch.LongTensor): Timestep indices of shape [batch_size]
        e (torch.Tensor): Noise samples of same shape as x0
        b (torch.Tensor): Beta schedule values - noise variance at each timestep
        keepdim (bool): Whether to keep dimensions when summing squared error
    
    Returns:
        torch.Tensor: Loss value (reduced to scalar if keepdim=False)
    
    Implementation Details:
        1. Computes alpha cumulative product based on provided beta schedule
        2. Adds noise to clean data according to diffusion timesteps
        3. Predicts noise using the model
        4. Calculates mean squared error between predicted and actual noise
    """
    # Calculate cumulative product of (1-beta) to get alpha_cumprod
    # These determine noise level at each timestep
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    
    # Forward diffusion process: add noise to clean data
    # x_t = sqrt(a)*x_0 + sqrt(1-a)*noise
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Predict noise using the model
    output = model(x, t.float())
    
    # Calculate squared error between predicted and actual noise
    if keepdim:
        # Return per-sample loss (summed across spatial dimensions)
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        # Return mean loss across batch
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def conditional_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          x_scale,
                          x_offset,
                          keepdim=False, p=0.1):
    """
    Conditional noise estimation loss with physics-informed guidance.
    
    This function implements the hybrid loss strategy that combines standard diffusion
    training with physics-based guidance. With probability p, it uses standard denoising
    score matching; otherwise, it incorporates physical gradient information as a condition.
    
    This dual approach allows the model to learn both from data statistics and physical laws,
    creating a physics-informed diffusion process.
    
    Args:
        model (nn.Module): The conditional denoising network that can accept physical gradients
        x0 (torch.Tensor): Clean data samples of shape [batch_size, channels, height, width]
        t (torch.LongTensor): Timestep indices of shape [batch_size]
        e (torch.Tensor): Noise samples of same shape as x0
        b (torch.Tensor): Beta schedule values
        x_scale (float): Scaling factor for denormalization of data
        x_offset (float): Offset value for denormalization of data
        keepdim (bool): Whether to keep dimensions when summing squared error
        p (float): Probability of using standard (non-physical) training. Default: 0.1
    
    Returns:
        torch.Tensor: Loss value (reduced to scalar if keepdim=False)
    
    Implementation Details:
        1. Same diffusion mechanics as standard loss
        2. Uses random sampling to determine whether to apply physical constraints
        3. When using physics, calculates vorticity residual and passes it to model
        4. Transforms between normalized model space and physical domain for correct gradient calculation
    """
    # Calculate alpha cumulative product (same as in standard loss)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    
    # Add noise to clean data following diffusion process
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # Random decision: standard training vs. physics-informed training
    flag = np.random.uniform(0, 1)
    if flag < p:
        # With probability p: use standard noise prediction (no physics)
        output = model(x, t.float())
    else:
        # With probability (1-p): use physics-informed prediction
        # 1. Transform normalized data back to physical domain
        # 2. Calculate physical residual gradient
        # 3. Scale gradient back to normalized domain
        # 4. Pass both noisy data and physical gradient to conditional model
        dx = voriticity_residual((x*x_scale + x_offset)) / x_scale
        output = model(x, t.float(), dx)
    
    # Calculate loss (same as standard version)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# Registry for selecting loss functions
loss_registry = {
    'simple': noise_estimation_loss,
    'conditional': conditional_noise_estimation_loss
}
