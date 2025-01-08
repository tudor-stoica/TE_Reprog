import numpy as np
import torch
import torch.nn as nn

# Computes the entropy of the input predictions.
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5  # Small value to avoid log(0)
    entropy = -input_ * torch.log(input_ + epsilon)  # Element-wise entropy computation
    entropy = torch.sum(entropy, dim=1)  # Sum across classes for each instance
    return entropy

# Gradient reversal hook function for gradient reversal layers
def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()  # Negate the gradient with scaling
    return fun1

# Conditional Domain Adversarial Network (CDAN) loss computation.
def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None, device="cpu"):
    # Extract softmax predictions and feature representations from the input list
    softmax_output = input_list[1].detach()  # Detach to avoid gradient computation
    feature = input_list[0]  # Features extracted by the backbone network

    # Compute the output of the adversarial network
    if random_layer is None:    # If no random layer is provided, combine features and softmax predictions using outer product
        # Compute the outer product between softmax outputs and feature representations
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        # Flatten the combined feature space and pass it through the adversarial network
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:   # If a random layer is provided, process the input list through it
        random_out = random_layer.forward([feature, softmax_output])
        # Pass the transformed features through the adversarial network
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    
    # Define domain classification targets (1 for source, 0 for target)
    batch_size = softmax_output.size(0) // 2  # Assume equal source and target batches
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)

    if entropy is not None:
        # Attach a gradient reversal hook for entropy scaling
        entropy.register_hook(grl_hook(coeff))
        
        # Compute entropy-based weights
        entropy = 1.0 + torch.exp(-entropy)  # Higher entropy results in higher weight
        
        # Create masks for source and target weights
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0  # Zero out target weights
        source_weight = entropy * source_mask

        target_mask = torch.ones_like(entropy)
        target_mask[:feature.size(0) // 2] = 0  # Zero out source weights
        target_weight = entropy * target_mask

        # Normalize weights for source and target
        weight = (
            source_weight / torch.sum(source_weight).detach().item()
            + target_weight / torch.sum(target_weight).detach().item()
        )
        
        # Compute weighted binary cross-entropy loss
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction="none")(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        # If no entropy is provided, compute standard binary cross-entropy loss
        return nn.BCELoss()(ad_out, dc_target)

# Domain Adversarial Neural Network (DANN) loss computation.
def DANN(features, ad_net, device="cpu"):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    return nn.BCELoss()(ad_out, dc_target)

# Cross entropy loss with label smoothing regularization.
class CrossEntropyLabelSmooth(nn.Module):
    """ Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, device="cpu", reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.device = device  # Add device parameter
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # Forward pass for computing the smoothed cross-entropy loss.
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        # Compute log probabilities
        log_probs = self.logsoftmax(inputs)
        
        # Convert labels to one-hot encoding
        targets = torch.zeros(log_probs.size(), device=self.device).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu and self.device != "cpu":
            targets = targets.to(self.device)
        
        # Apply label smoothing
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        # Compute the loss
        loss = (-targets * log_probs).sum(dim=1)
        return loss.mean() if self.reduction else loss