import torch
import torch.nn as nn

class ReprogrammableLayer(nn.Module):
    def __init__(self, input_shape, source_time_len, target_time_len, device="cpu"):
        super(ReprogrammableLayer, self).__init__()
        self.source_time_len = source_time_len
        self.target_time_len = target_time_len
        self.features = input_shape[2]  # Number of features
        self.device = device

        # Trainable reprogramming matrix (θ)
        self.theta = nn.Parameter(
            torch.randn(source_time_len - target_time_len, self.features, device=self.device),
            requires_grad=True
        )

        # Mask: First target_time_len rows are 0, rest are 1
        self.mask = torch.zeros((source_time_len, self.features), device=self.device)
        self.mask[target_time_len:, :] = 1.0  # Apply ones to rows beyond target_time_len

    def forward(self, target_input):
        # Expect target_input shape: [batch_size, channels, target_time_len, features]
        batch_size, channels, target_time_len, features = target_input.shape  

        # Pad target input with zeros to match source_time_len
        padding = torch.zeros(batch_size, channels, self.source_time_len - self.target_time_len, self.features, device=self.device)
        padded_input = torch.cat((target_input, padding), dim=2)  # Concatenate along the time dimension (dim=2)

        # Expand θ to match the mask dimensions
        expanded_theta = torch.zeros(self.source_time_len, self.features, device=self.device)
        expanded_theta[self.target_time_len:, :] = self.theta  # Fill only the trainable part

        # Apply mask to θ
        masked_theta = (expanded_theta * self.mask).unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, source_time_len, features]
        masked_theta = masked_theta.expand(batch_size, channels, -1, -1)  # Shape: [batch_size, channels, source_time_len, features]

        # Add masked θ to the padded input
        reprogrammed_input = padded_input + masked_theta

        return reprogrammed_input

# Map target classes to source classes.
def map_target_to_source_classes(num_source_classes, num_target_classes):
    # Calculate the number of source classes per target class
    source_classes_per_target = num_source_classes // num_target_classes

    # Initialize the mapping
    target_to_source_map = []

    # Create the mapping
    for target_class in range(num_target_classes):
        # Assign the next block of source classes to this target class
        start_idx = target_class * source_classes_per_target
        end_idx = start_idx + source_classes_per_target
        target_to_source_map.extend(range(start_idx, end_idx))

    # Convert to tensor for use in PyTorch
    target_to_source_map = torch.tensor(target_to_source_map[:num_target_classes], dtype=torch.long)

    return target_to_source_map