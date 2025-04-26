import torch.nn as nn

class EMAHelper(object):
    """
    Exponential Moving Average Helper class for model parameter smoothing.
    
    In diffusion models, EMA plays a critical role by:
    1. Stabilizing training by reducing parameter oscillations
    2. Improving generalization and generation quality
    3. Creating smoother, more physically consistent fluid fields
    4. Reducing overfitting to recent batches
    
    Best practices:
    - Use high decay rates (0.995-0.9999) for diffusion models
    - Call update() after EACH optimizer.step()
    - Use EMA parameters for inference/evaluation, not for training
    - Always save EMA state in checkpoints for proper resumption
    """
    def __init__(self, mu=0.999):
        """
        Initialize the EMA helper with a decay rate.
        
        Args:
            mu (float): Decay rate for EMA, typically very close to 1.
                - Higher values (0.9999) result in more stable but slower adaptation
                - Lower values (0.99) adapt faster but may be less stable
                - For fluid field reconstruction, 0.999-0.9999 is recommended
        """
        self.mu = mu  # EMA decay rate (momentum)
        self.shadow = {}  # Dictionary to store EMA parameters
    
    def register(self, module):
        """
        Register module parameters to initialize shadow parameters.
        
        This should be called at the beginning of training, right after
        model initialization and before any training steps.
        
        Args:
            module: PyTorch model whose parameters need to be tracked
            
        Best practice:
            Always call this function immediately after model initialization:
            ```
            model = Model()
            ema_helper = EMAHelper(mu=0.999)
            ema_helper.register(model)
            ```
        """
        if isinstance(module, nn.DataParallel):
            module = module.module  # Unwrap DataParallel to access the actual model
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()  # Create initial copy of each trainable parameter
    
    def update(self, module):
        """
        Update shadow parameters using EMA formula.
        
        This is the core EMA functionality that performs the smoothing operation.
        It should be called after each optimization step.
        
        Formula: shadow = (1-mu)*param + mu*shadow
        
        Args:
            module: PyTorch model whose current parameters will update the EMA
            
        Best practice:
            Call immediately after optimizer.step() in the training loop:
            ```
            optimizer.step()
            if ema_helper is not None:
                ema_helper.update(model)
            ```
        """
        if isinstance(module, nn.DataParallel):
            module = module.module  # Unwrap DataParallel to access the actual model
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data  # EMA update formula
    
    def ema(self, module):
        """
        Apply EMA parameters to the model (modifies the model in-place).
        
        This replaces the model's current parameters with the EMA parameters.
        Used before evaluation/inference but not during training.
        
        Args:
            module: PyTorch model whose parameters will be replaced with EMA values
            
        Best practice:
            Use before evaluation/inference, then restore original parameters for training:
            ```
            # Before evaluation
            ema_helper.ema(model)
            evaluate(model)
            # Restore original parameters by loading saved state
            model.load_state_dict(original_state_dict)
            ```
        """
        if isinstance(module, nn.DataParallel):
            module = module.module  # Unwrap DataParallel to access the actual model
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)  # Replace model parameters with EMA parameters
    
    def ema_copy(self, module):
        """
        Create a new model instance with EMA parameters.
        
        Unlike ema(), this doesn't modify the original model but returns
        a new model instance with EMA parameters. This is safer when you
        need to keep the original model intact.
        
        Args:
            module: Original PyTorch model
            
        Returns:
            A new model instance with EMA parameters
            
        Best practice:
            Use for inference while keeping training model untouched:
            ```
            # For evaluation/inference
            ema_model = ema_helper.ema_copy(model)
            results = evaluate(ema_model)
            # Original model is unchanged
            ```
        """
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)  # Create new instance of same model type
            module_copy.load_state_dict(inner_module.state_dict())  # Copy current parameters
            module_copy = nn.DataParallel(module_copy)  # Wrap in DataParallel if original was wrapped
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)  # Alternative but may be less efficient
        self.ema(module_copy)  # Apply EMA parameters to the new model
        return module_copy
    
    def state_dict(self):
        """
        Return the EMA shadow parameters for saving checkpoints.
        
        Returns:
            dict: The shadow parameters dictionary
            
        Best practice:
            Always include EMA state when saving checkpoints:
            ```
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
                ema_helper.state_dict()  # Save EMA state
            ]
            torch.save(states, checkpoint_path)
            ```
        """
        return self.shadow
    
    def load_state_dict(self, state_dict):
        """
        Load EMA shadow parameters from a previously saved state.
        
        Critical for resuming training with consistent EMA tracking.
        
        Args:
            state_dict (dict): Previously saved shadow parameters
            
        Best practice:
            When resuming training, always restore EMA state:
            ```
            states = torch.load(checkpoint_path)
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            epoch = states[2]
            step = states[3]
            ema_helper.load_state_dict(states[4])  # Restore EMA state
            ```
        """
        self.shadow = state_dict
