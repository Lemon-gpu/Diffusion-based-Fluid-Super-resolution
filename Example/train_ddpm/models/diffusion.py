import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    # Convert timestep value(s) to a more expressive feature representation using sinusoidal encoding
    # Similar to positional encoding in transformers, but adapted for diffusion models
    # BEST PRACTICE: Using sinusoidal embeddings helps the model distinguish different timesteps in the diffusion process
    assert len(timesteps.shape) == 1  # Ensure timesteps is a 1D tensor

    half_dim = embedding_dim // 2  # Use half the dimensions for sine, half for cosine
    emb = math.log(10000) / (half_dim - 1)  # Scale factor for the embedding
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  # Create exponentially decreasing frequencies
    emb = emb.to(device=timesteps.device)  # Move to the same device as timesteps
    emb = timesteps.float()[:, None] * emb[None, :]  # Outer product of timesteps and frequencies
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # Concatenate sine and cosine components
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  # Handle odd embedding dimensions with padding
    return emb


def nonlinearity(x):
    # swish
    # Implementation of the Swish activation function: x * sigmoid(x)
    # BEST PRACTICE: Swish often performs better than ReLU in deep networks, especially for diffusion models
    # It provides smoother gradients and helps with training stability
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    # Creates a GroupNorm normalization layer
    # BEST PRACTICE: GroupNorm is preferred over BatchNorm for diffusion models as:
    # 1. It's more stable with small/varying batch sizes 
    # 2. Has consistent behavior during training and inference
    # 3. Works better for the spatial data in fluid simulations
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)
    # return torch.nn.GroupNorm(num_groups=20, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # Upsampling module that doubles the spatial resolution of feature maps
        # Used in the decoder part of the U-Net architecture
        # BEST PRACTICE: Optional convolutional layer after upsampling helps reduce checkerboard artifacts
        super().__init__()
        self.with_conv = with_conv  # Boolean flag to determine if a convolution is applied after upsampling
        if self.with_conv:
            # If with_conv is True, apply a 3x3 convolution after upsampling
            # Circular padding is crucial for fluid simulations to ensure spatial continuity at boundaries
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')

    def forward(self, x):
        # First, upsample the input using nearest neighbor interpolation to double spatial dimensions
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            # If with_conv is True, apply the convolution to smooth the upsampled features
            # This reduces aliasing artifacts and helps maintain fluid dynamics consistency
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # Downsampling module that halves the spatial resolution of feature maps
        # Used in the encoder part of the U-Net architecture
        # BEST PRACTICE: Convolutional downsampling preserves more information than pooling
        super().__init__()
        self.with_conv = with_conv  # Boolean flag to determine downsampling method
        if self.with_conv:
            # If with_conv is True, use strided convolution for downsampling
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,  # Stride of 2 halves the spatial dimensions
                                        padding=0)  # No padding here because we'll add it manually

    def forward(self, x):
        if self.with_conv:
            # Manual asymmetric padding specifically designed for fluid simulation data
            # Pad right and bottom by 1 to ensure proper downsampling with circular boundary
            pad = (0, 1, 0, 1)  # (left, right, top, bottom)
            x = torch.nn.functional.pad(x, pad, mode="circular")  # Circular padding ensures fluid continuity
            x = self.conv(x)  # Apply strided convolution for downsampling
        else:
            # If not using convolution, use average pooling instead
            # Average pooling preserves more information than max pooling for fluid dynamics
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        # Residual block with time embedding conditioning for the diffusion model
        # BEST PRACTICE: Residual connections are crucial for training deep networks
        # The time embedding allows the block to adapt its behavior based on the diffusion timestep
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels  # Default output channels to input channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut  # Whether to use convolution in the residual path

        # First normalization and convolution path
        self.norm1 = Normalize(in_channels)  # Group normalization before first convolution
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode='circular')  # Circular padding maintains fluid continuity
        
        # Time embedding projection - allows the diffusion timestep to influence the residual block
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)  # Projects time embedding to feature dimensions
        
        # Second normalization, dropout, and convolution path
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)  # Dropout for regularization
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode='circular')
        
        # Skip connection (residual path) handling
        if self.in_channels != self.out_channels:
            # Handle different input and output channel dimensions
            if self.use_conv_shortcut:
                # Use 3x3 convolution for the skip connection
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     padding_mode='circular')
            else:
                # Use 1x1 convolution for the skip connection (more efficient)
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x  # Store input for the residual connection
        
        # First conv block
        h = self.norm1(h)  # Normalize
        h = nonlinearity(h)  # Apply Swish activation
        h = self.conv1(h)  # First convolution

        # Add time embedding influence
        # This is a critical part that allows the block to behave differently at different timesteps
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]  # Broadcast time embedding to spatial dimensions

        # Second conv block
        h = self.norm2(h)  # Second normalization
        h = nonlinearity(h)  # Second activation
        h = self.dropout(h)  # Apply dropout for regularization
        h = self.conv2(h)  # Second convolution

        # Handle skip connection for different channel dimensions
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)  # 3x3 convolution for skip connection
            else:
                x = self.nin_shortcut(x)  # 1x1 convolution for skip connection

        return x+h  # Add the residual connection to the processed features


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        # Self-attention block for capturing long-range dependencies in the fluid field
        # BEST PRACTICE: Attention helps model global context and coherent structures in fluid simulations
        # This is particularly important for correctly representing vortices and other global flow patterns
        super().__init__()
        self.in_channels = in_channels

        # Normalization before attention operations
        self.norm = Normalize(in_channels)
        
        # Query projection
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,  # 1x1 convolution for projection
                                 stride=1,
                                 padding=0)
        # Key projection
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,  # 1x1 convolution for projection
                                 stride=1,
                                 padding=0)
        # Value projection
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,  # 1x1 convolution for projection
                                 stride=1,
                                 padding=0)
        # Output projection
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,  # 1x1 convolution for projection
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x  # Store input for the residual connection
        h_ = self.norm(h_)  # Normalize input
        
        # Project to query, key, and value
        q = self.q(h_)  # Query projection
        k = self.k(h_)  # Key projection
        v = self.v(h_)  # Value projection

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)  # Reshape query to [batch, channels, height*width]
        q = q.permute(0, 2, 1)    # Transpose to [batch, height*width, channels]
        k = k.reshape(b, c, h*w)  # Reshape key to [batch, channels, height*width]
        
        # Calculate attention weights through matrix multiplication of query and key
        w_ = torch.bmm(q, k)      # [batch, height*width, height*width] attention map
        # Scale attention weights by square root of channels (attention scaling)
        w_ = w_ * (int(c)**(-0.5))
        # Apply softmax to get attention probabilities
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)  # Reshape value to [batch, channels, height*width]
        w_ = w_.permute(0, 2, 1)  # Transpose attention weights
        # Apply attention weights to values
        h_ = torch.bmm(v, w_)  # Matrix multiply value and attention weights
        h_ = h_.reshape(b, c, h, w)  # Reshape back to original spatial dimensions

        # Project to output and apply residual connection
        h_ = self.proj_out(h_)  # Output projection

        return x+h_  # Add the residual connection to the attention output


class Model(nn.Module):
    def __init__(self, config):
        # Main diffusion model implementation - a time-conditional U-Net architecture
        # This is the unconditional model used for basic diffusion tasks
        # BEST PRACTICE: This architecture balances capacity and efficiency for fluid simulation tasks
        super().__init__()
        self.config = config
        
        # Extract configuration parameters
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks  # Number of residual blocks per resolution level
        attn_resolutions = config.model.attn_resolutions  # Resolutions at which to apply attention
        dropout = config.model.dropout  # Dropout rate for regularization
        in_channels = config.model.in_channels  # Input channels (fluid field components)
        resolution = config.data.image_size  # Spatial resolution of the input
        resamp_with_conv = config.model.resamp_with_conv  # Whether to use convolution for resampling
        num_timesteps = config.diffusion.num_diffusion_timesteps  # Number of diffusion timesteps
        
        # For Bayesian models, add a learnable logvar parameter for each timestep
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        # Store key architecture parameters
        self.ch = ch  # Base channel multiplier
        self.temb_ch = self.ch*4  # Time embedding channels (4x base channels)
        self.num_resolutions = len(ch_mult)  # Number of resolution levels in U-Net
        self.num_res_blocks = num_res_blocks  # Number of residual blocks per level
        self.resolution = resolution  # Input resolution
        self.in_channels = in_channels  # Number of input channels

        # timestep embedding - converts scalar timestep to feature vector
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),  # First linear layer for time embedding
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),  # Second linear layer for time embedding
        ])

        # downsampling - encoder part of U-Net
        # Initial convolution to map input fluid field to feature space
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode='circular')  # Circular padding for fluid continuity

        curr_res = resolution  # Track current resolution through the network
        in_ch_mult = (1,)+ch_mult  # Channel multipliers including the input level
        self.down = nn.ModuleList()  # List to store all downsampling blocks
        block_in = None  # Will track number of input channels to each block
        
        # Build the encoder (downsampling path)
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()  # Residual blocks at current resolution
            attn = nn.ModuleList()  # Attention blocks at current resolution
            block_in = ch*in_ch_mult[i_level]  # Input channels to this resolution level
            block_out = ch*ch_mult[i_level]  # Output channels from this resolution level
            
            # Add the specified number of residual blocks at this resolution
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out  # Update input channels for next block
                
                # Add attention block if current resolution requires it
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            # Create a module for this resolution level
            down = nn.Module()
            down.block = block  # Residual blocks
            down.attn = attn  # Attention blocks
            
            # Add downsampling except for the last resolution level
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2  # Halve the resolution
            
            self.down.append(down)  # Add this resolution level to the encoder

        # middle - bottleneck of U-Net
        self.mid = nn.Module()
        # First residual block at lowest resolution
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # Attention block at lowest resolution for global context
        self.mid.attn_1 = AttnBlock(block_in)
        # Second residual block at lowest resolution
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling - decoder part of U-Net
        self.up = nn.ModuleList()  # List to store all upsampling blocks
        
        # Build the decoder (upsampling path) - mirror of encoder
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()  # Residual blocks at current resolution
            attn = nn.ModuleList()  # Attention blocks at current resolution
            block_out = ch*ch_mult[i_level]  # Output channels for this resolution level
            skip_in = ch*ch_mult[i_level]  # Skip connection input channels
            
            # Add one more residual block than in encoder for asymmetric U-Net
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]  # Adjust skip connection channels for last block
                
                # Residual block combining upsampled features and skip connection
                block.append(ResnetBlock(in_channels=block_in+skip_in,  # Concatenated current and skip features
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out  # Update input channels for next block
                
                # Add attention block if current resolution requires it
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            # Create a module for this resolution level
            up = nn.Module()
            up.block = block  # Residual blocks
            up.attn = attn  # Attention blocks
            
            # Add upsampling except for the first (highest) resolution level
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2  # Double the resolution
            
            # Insert at beginning to preserve resolution order (lowest to highest)
            self.up.insert(0, up)  # prepend to get consistent order

        # end - final layers to produce output
        self.norm_out = Normalize(block_in)  # Final normalization
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,  # Map to output channels (typically matching input)
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')  # Circular padding for fluid continuity

    def forward(self, x, t):
        # Forward pass of the diffusion model
        # x: fluid field at a noisy timestep [batch, channels, height, width]
        # t: diffusion timesteps [batch]
        assert x.shape[2] == x.shape[3] == self.resolution  # Verify input resolution matches model

        # timestep embedding - convert scalar timesteps to feature vectors
        temb = get_timestep_embedding(t, self.ch)  # Initial sinusoidal embedding
        # Process through linear layers with nonlinearity
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling - encoder path
        hs = [self.conv_in(x)]  # Initial features from input convolution
        # Process through all encoder levels
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # Pass through residual block with time conditioning
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # Add attention if defined for this block
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)  # Store for skip connections
            # Downsample except at last level
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle - bottleneck processing
        h = hs[-1]  # Features at lowest resolution
        h = self.mid.block_1(h, temb)  # First residual block
        h = self.mid.attn_1(h)  # Self-attention for global context
        h = self.mid.block_2(h, temb)  # Second residual block

        # upsampling - decoder path with skip connections
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                # Concatenate current features with skip connection
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)  # Skip connection via concatenation
                # Add attention if defined for this block
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # Upsample except at highest resolution (i_level=0)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end - final processing to output
        h = self.norm_out(h)  # Normalize
        h = nonlinearity(h)  # Activate
        h = self.conv_out(h)  # Convert to output channels
        return h  # Return predicted noise or clean fluid field (depending on parameterization)

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        # Spectral convolution layer that operates in the Fourier domain
        # BEST PRACTICE: For fluid dynamics, spectral methods are often more accurate as they naturally
        # handle the wave-like behavior and preserve physical conservation laws
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2  # Number of Fourier modes in second dimension

        # Initialize Fourier weights with a scaling factor
        self.scale = (1 / (in_channels * out_channels))  # Scaling factor for initialization
        # Complex weights for the Fourier space transformation
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # Performs multiplication in complex Fourier space
        # Efficient implementation using Einstein summation
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # Forward pass implementing spectral convolution
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)  # Real-to-complex FFT in 2D

        # Multiply relevant Fourier modes
        # Initialize output tensor in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply lower frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # Multiply higher frequencies (for 2D real FFT, only need to handle positive and negative frequencies in first dimension)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space using inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        # Fourier Neural Operator for 2D problems
        # Powerful for modeling fluid dynamics as it can learn complex spatial patterns
        # BEST PRACTICE: FNOs are particularly effective for PDEs like those governing fluid mechanics
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1  # Number of Fourier modes to keep in first dimension
        self.modes2 = modes2  # Number of Fourier modes to keep in second dimension
        self.width = width  # Hidden channel dimension
        self.padding = 2  # pad the domain if input is non-periodic
        
        # Input embedding layer - maps 12 input features to hidden width
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # Four layers of spectral convolutions (Fourier layers)
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        
        # Convolutional layers to be applied alongside spectral convolutions
        self.w0 = nn.Conv2d(self.width, self.width, 1)  # 1x1 convolution for pointwise mixing
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Batch normalization layers for stable training
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        # Output layers - project from hidden width to output space
        self.fc1 = nn.Linear(self.width, 128)  # First output projection
        self.fc2 = nn.Linear(128, 1)  # Final output layer to single value (next timestep)

    def forward(self, x):
        # Forward pass of the Fourier Neural Operator
        # Generate mesh grid to provide positional information
        grid = self.get_grid(x.shape, x.device)  # Get coordinate grid
        x = torch.cat((x, grid), dim=-1)  # Concatenate input with coordinate information
        
        # Initial linear embedding
        x = self.fc0(x)  # [batch, x, y, width]
        x = x.permute(0, 3, 1, 2)  # Reorder to [batch, width, x, y] for convolution operations
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        # Layer 0: Spectral + spatial convolution with activation
        x1 = self.conv0(x)  # Spectral convolution (Fourier domain)
        x2 = self.w0(x)  # Spatial convolution (physical domain)
        x = x1 + x2  # Combine spectral and spatial paths
        x = F.gelu(x)  # GELU activation function

        # Layer 1: Spectral + spatial convolution with activation
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2: Spectral + spatial convolution with activation
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 3: Spectral + spatial convolution with activation
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # Output processing
        x = x.permute(0, 2, 3, 1)  # Reorder to [batch, x, y, width]
        x = self.fc1(x)  # First output projection
        x = F.gelu(x)  # Final activation
        x = self.fc2(x)  # Project to output dimension
        return x

    def get_grid(self, shape, device):
        # Generate a mesh grid of normalized coordinates
        # This provides positional information to the model, which is essential for PDEs
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # Create x-coordinates, normalized to [0,1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        # Create y-coordinates, normalized to [0,1]
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        # Concatenate x and y coordinates and move to device
        return torch.cat((gridx, gridy), dim=-1).to(device)


class ConditionalModel(nn.Module):
    def __init__(self, config):
        # Conditional diffusion model with physics-informed gradient embedding
        # This extends the base Model to incorporate physical gradient information
        # BEST PRACTICE: Conditioning on physical gradients improves accuracy and stability for fluid simulations
        super().__init__()
        self.config = config
        
        # Extract configuration parameters (same as base Model)
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        # For Bayesian models, add learnable logvar parameter
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        # Store key architecture parameters
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding - same as base Model
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # gradient embedding - processes physical gradient information
        # This is unique to the conditional model, allowing it to incorporate physics-based guidance
        self.emb_conv = nn.Sequential(
            torch.nn.Conv2d(in_channels, self.ch, kernel_size=1, stride=1, padding=0),  # 1x1 convolution
            nn.GELU(),  # GELU activation
            torch.nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, padding_mode='circular')  # 3x3 convolution
        )

        # downsampling - encoder path
        # Input convolution - same as base Model
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                        padding=1, padding_mode='circular')

        # Combine input features with gradient embedding using 1x1 convolution
        self.combine_conv = torch.nn.Conv2d(self.ch*2, self.ch, kernel_size=1, stride=1, padding=0)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        
        # Build encoder (downsampling path) - same structure as base Model
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle - bottleneck (same as base Model)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling - decoder path (same as base Model)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end - final layers (same as base Model)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')
        # self.spectral_regressor1 = SpectralConv2d_fast(ch, ch, 12, 12)
        # self.spectral_regressor2 = SpectralConv2d_fast(ch, ch, 12, 12)
        # self.final_norm = Normalize(ch)
        # self.to_out = torch.nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x, t, dx=None):
        # Forward pass with conditional physical gradient information
        # x: input noisy fluid field [batch, channels, height, width]
        # t: diffusion timesteps [batch]
        # dx: physical gradient conditioning information [batch, channels, height, width]
        
        # dx is the physical gradient, working as context embedding
        assert x.shape[2] == x.shape[3] == self.resolution  # Verify input resolution

        # timestep embedding - same as base Model
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # Process input and gradient embedding
        x = self.conv_in(x)  # Process input fluid field
        if dx is not None:
            # If gradient information is provided, embed it
            cond_emb = self.emb_conv(dx)  # Process physical gradient
        else:
            # Otherwise use zeros (unconditional generation)
            cond_emb = torch.zeros_like(x)
        
        # Concatenate input features with gradient embedding
        x = torch.cat((x, cond_emb), dim=1)
    
        # downsampling with combined features
        hs = [self.combine_conv(x)]  # Initial combined features

        # Encoder path - same structure as base Model
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle - bottleneck processing (same as base Model)
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling - decoder path (same as base Model)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end - final processing to output (same as base Model)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
