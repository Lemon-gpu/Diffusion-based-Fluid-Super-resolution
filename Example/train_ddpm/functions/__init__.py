##############################################################################
# Optimizer Configuration Module for Physics-informed Diffusion Models
##############################################################################
# 
# This module provides optimizer configuration functionality for the physics-informed
# diffusion model training process. It serves as a critical component in balancing 
# the multi-objective optimization required in physics-guided training, where both
# data-driven loss and physical constraints must be satisfied simultaneously.
#
# In physics-informed diffusion models for fluid super-resolution, optimizer selection
# and configuration are particularly important because:
#   1. The model needs to balance between data fidelity and physical consistency
#   2. Physical gradients may have different scales compared to standard noise prediction
#   3. Training stability is crucial when incorporating physical PDEs as constraints
#   4. Different optimizers may handle the trade-offs between physical and data terms differently
#
# Best practices for optimizer selection in physics-informed diffusion models:
#   - Adam optimizer generally performs best with slightly lower learning rates (1e-4 to 2e-4)
#   - Higher beta1 values (0.9) help stabilize training with physical constraints
#   - Weight decay should be carefully tuned to prevent overfitting without compromising
#     physical accuracy
#   - For turbulent flow fields with complex physical dynamics, amsgrad=True can help
#     with convergence

import torch.optim as optim


def get_optimizer(config, parameters):
    """
    Creates and returns an optimizer based on configuration parameters.
    
    This function instantiates the appropriate optimizer according to the provided
    configuration. It serves as a centralized optimizer factory for the training
    process, ensuring consistent optimizer creation throughout the codebase.
    
    For physics-informed diffusion models, optimizer selection significantly impacts
    training stability and convergence, particularly when balancing between data-driven
    loss terms and physics-based constraint terms.
    
    Args:
        config: Configuration object containing optimizer settings.
               Should have attributes like config.optim.optimizer (str),
               config.optim.lr (float), etc.
        parameters: Model parameters to optimize (typically model.parameters())
    
    Returns:
        torch.optim.Optimizer: Configured optimizer instance
        
    Raises:
        NotImplementedError: If the specified optimizer is not supported
        
    Best practices:
        - For standard diffusion model training: Adam with lr=2e-4
        - For physics-guided training: Adam with lr=1e-4 to 2e-4
        - When physical constraints dominate early training: Consider RMSProp
        - For fine-tuning: SGD with lower learning rates
    """
    # Handle Adam optimizer case - preferred for diffusion models due to adaptive learning
    # rates and momentum, which help navigate the complex loss landscape of physics-guided
    # diffusion models
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    # Handle RMSProp case - can be useful when physical gradients have high variance
    # as it normalizes gradients by their recent magnitude
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    # Handle SGD case - useful for fine-tuning or when very stable convergence is required
    # Fixed momentum value of 0.9 provides good balance between stability and training speed
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    # Raise exception if optimizer type is not recognized
    # This ensures that only supported optimizers are used, preventing silent errors
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))
