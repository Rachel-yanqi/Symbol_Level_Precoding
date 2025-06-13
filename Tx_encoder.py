import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
# from generate_channel import generate_correlated_CSI
torch.manual_seed(42)

"""
Adaptive Modulation
Robustness test - time varying CSI 
"""
class SpatialAttention(nn.Module):
    """
    Find most correlated featrure spatially, CSI input
    """
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid()    # Attention weights between 0 and 1
        )
    
    def forward(self, x):
        att_weights = self.attention(x)     #(B, 1, N, N)
        return x * att_weights

class ChannelAttention(nn.Module):
    """
    Channel Attention: Learns WHAT features are important
    For UAV networks: Which channel characteristics (H_real, H_imag, etc.) matter most
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # Global pooling to get channel-wise statistics
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooled features
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        
    def forward(self, x):
        # x shape: (B, channels, H, W)
        batch_size, channels, _, _ = x.size()
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        avg_pooled = self.global_avg_pool(x).view(batch_size, channels)
        max_pooled = self.global_max_pool(x).view(batch_size, channels)
        
        # Generate channel attention weights
        avg_weights = self.mlp(avg_pooled)  # (B, C)
        max_weights = self.mlp(max_pooled)  # (B, C)
        
        # Combine and apply sigmoid
        channel_weights = torch.sigmoid(avg_weights + max_weights)  # (B, C)
        
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        channel_weights = channel_weights.view(batch_size, channels, 1, 1)
        
        # Apply channel attention
        attended_features = x * channel_weights
        
        return attended_features, channel_weights


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate = 0.1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        # If input and output channels don't match, use a 1x1 conv to match dimensions
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        residual = self.shortcut(x)
        return self.dropout(self.conv_block(x)) + residual

class SymbolLevelPrecodingNetwork(nn.Module):
    def __init__(self, num_users=4, constellation_size=16):
        super().__init__()

        if constellation_size == 4:
            # 4-QAM Constellation Points
            self.constellation_points = torch.tensor([
                -1 - 1j, -1 + 1j,
                1 - 1j, 1 + 1j
            ] / np.sqrt(2), dtype=torch.complex64)

        elif constellation_size == 8:
            self.constellation_points = torch.tensor([
                    1 + 0j,
                    np.sqrt(2)/2 + 1j*np.sqrt(2)/2,
                    0 + 1j,
                    -np.sqrt(2)/2 + 1j*np.sqrt(2)/2,
                    -1 + 0j,
                    -np.sqrt(2)/2 - 1j*np.sqrt(2)/2,
                    0 - 1j,
                    np.sqrt(2)/2 - 1j*np.sqrt(2)/2
                ], dtype=torch.complex64)

        elif constellation_size == 16:
            self.constellation_points = torch.tensor([
                    -3-3j, -3-1j, -3+1j, -3+3j,
                    -1-3j, -1-1j, -1+1j, -1+3j,
                    1-3j, 1-1j, 1+1j, 1+3j,
                    3-3j, 3-1j, 3+1j, 3+3j
                ] / np.sqrt(10), dtype=torch.complex64)

        self.num_users = num_users
        self.constellation_size = constellation_size

        # Learnable SNR-to-power mapping
        self.snr_to_power = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Neural network layers for joint channel state and symbol processing
        # Symbol embedding creates spatial features
        self.symbol_to_spatial = nn.Sequential(
            nn.Linear(constellation_size, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, num_users * num_users)  # Match channel matrix size
        )

        # Input channels: 2 (H_real, H_imag) + num_users (symbol spatial maps)  1 (snr_map)
        input_channels = 2 + 2 + 1 # 5

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1, padding=0), 
            nn.GroupNorm(8, 32),  # GroupNorm often works better than BatchNorm
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(32)
        )
        # Residual blocks
        # Residual blocks with improved design
        self.res_blocks = nn.ModuleList([
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192),
            ResidualBlock(192, 256)
        ])
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 128)
        # Deepen network
        self.res_block4 = ResidualBlock(128, 192)
        self.res_block5 = ResidualBlock(192, 256)
        self.res_block6 = ResidualBlock(256, 256)

        self.spatial_feature_extractor = SpatialAttention(256)       
        self.channel_feature_extractor = ChannelAttention(256)

        # Consider direct precoding
        # Shape: (num_users, num_users, 2)
        fc_output_size = num_users * num_users * 2

        self.precoding_generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*self.num_users*self.num_users, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            # nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, fc_output_size)  # Real and imag parts
        )

    def symbol_embedding(self, symbols):
        """
        Convert constellation symbols by one-hot encoding
        Args:
            symbol: generated symbols (batch_size, num_users)
        Returns:
            embeded_symbols: one-hot embedded symbols (batch_size, num_users, constellation_size)
        """
        batch_size = symbols.shape[0]
        symbol_embeddings = torch.zeros(batch_size, self.num_users, self.constellation_size,
                                      device=symbols.device)
        
        for b in range(batch_size):
            for u in range(self.num_users):
                # Find closest constellation point
                distances = torch.abs(self.constellation_points - symbols[b, u])
                closest_idx = torch.argmin(distances)
                symbol_embeddings[b, u, closest_idx] = 1.0
                
        return symbol_embeddings

    def create_snr_map(self, snr_per_user):
        """Create spatial SNR map for convolutional processing"""
        batch_size = snr_per_user.shape[0]
        snr_map = torch.zeros(batch_size, 1, self.num_users, self.num_users)

        # Create spatial representation of SNR
        for i in range(self.num_users):
            if i < self.num_users:
                snr_map[:, 0, i, i] = snr_per_user[:, i]

        return snr_map

    def forward(self, channel_state_info, symbols, snr_per_user):
        """
        Generate symbol-level precoding matrix based on channel state information and symbols

        Args:
            channel_state_info: Input channel state information (batch_size, num_users, num_users)
            symbols: Symbol vector of shape (batch_size, num_users)
            channel_norm: Channel norm of shape (batch_size, 1, 1)
            snr_per_user: List of SNR values for each user (batch_size, num_users)
        Returns:
            torch.Tensor: Precoding matrix of shape (batch_size, num_users, num_users)
        """
        batch_size = channel_state_info.shape[0]

        # Convert channel info to real representation
        if torch.is_complex(channel_state_info):
            h_real = channel_state_info.real
            h_imag = channel_state_info.imag
        else:
            # Handle case where channel is already in real format
            h_real, h_imag = channel_state_info[:, 0], channel_state_info[:, 1]

        # Embed symbols 
        symbol_embeddings = self.symbol_embedding(symbols)
        # Convert each user's symbol embedding to spatial map
        symbol_maps = []
        for u in range(self.num_users):
            user_embedding = symbol_embeddings[:, u, :]  # (batch, constellation_size)
            spatial_map = self.symbol_to_spatial(user_embedding)  # (batch, users*users)
            spatial_map = spatial_map.view(batch_size, 1, self.num_users, self.num_users)
            symbol_maps.append(spatial_map)
        symbol_maps = torch.cat(symbol_maps, dim=1)     # (batch, num_users, num_users, num_users)
        # Expand symbols to match channel dimensions
        s_real, s_imag = symbols.real, symbols.imag
        s_real_expanded = s_real.unsqueeze(2).expand(-1, -1, self.num_users).unsqueeze(1)
        s_imag_expanded = s_imag.unsqueeze(2).expand(-1, -1, self.num_users).unsqueeze(1)

        # Create input tensor with channel and symbol information
        # Reshape to match expected input dimensions for Conv2d
        h_real = h_real.view(batch_size, 1, self.num_users, self.num_users)
        h_imag = h_imag.view(batch_size, 1, self.num_users, self.num_users)


        # Convert snr_per_user to snr_map[i, i] = snr （batch_size, 1, num_users, num_users)
        snr_map = self.create_snr_map(snr_per_user)

        # Concatenate along channel dimension
        # x = torch.cat([h_real, h_imag, symbol_maps, snr_map], dim=1)
        x = torch.cat([h_real, h_imag, s_real_expanded, s_imag_expanded, snr_map], dim=1)

        # Feature extraction
        x = self.initial_conv(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.spatial_feature_extractor(x)
        x, _ = self.channel_feature_extractor(x)


        # Generate precoding matrices for all possible symbol combinations
        precoding_vector = self.precoding_generator(x)

        # Reshape the output to get individual precoding matrices
        half_elements = precoding_vector.shape[1] // 2

        precoding_real = precoding_vector[:, :half_elements].view(
            batch_size, self.num_users, self.num_users
        )
        precoding_imag = precoding_vector[:, half_elements:].view(
            batch_size, self.num_users, self.num_users
        )


        # Combine real and imaginary parts
        precoding_matrix = torch.complex(precoding_real, precoding_imag)

        # Soft power constraint based on SNR
        if snr_per_user.dtype == torch.int64:
            snr_per_user = snr_per_user.float()
        avg_snr = torch.mean(snr_per_user, dim=1, keepdim=True).float()
        power_target = 0.5 + 0.5 * self.snr_to_power(avg_snr)  # Range: [0.2, 1.0]
        
        # Normalize to learned power target
        current_power = torch.mean(torch.abs(precoding_matrix) ** 2, dim=(1, 2))
        scaling = torch.sqrt(power_target.squeeze(-1) / (current_power + 1e-8))  # Shape: (64,)
        scaling = scaling.unsqueeze(-1).unsqueeze(-1)  # Shape: (64, 1, 1) for broadcasting
        
        # Apply scaling but allow some flexibility
        # precoding_matrix = precoding_matrix * (0.8 * scaling + 0.2)
        precoding_matrix = precoding_matrix * torch.sqrt(scaling)

        # Apply power constraint
        # precoding_matrix = self.normalize_power(precoding_matrix)

        return precoding_matrix


    def normalize_power(self, precoding_matrix, power_constraint=2.0):
        """
        Normalize precoding matrix to satisfy power constraint

        Args:
            precoding_matrix: Precoding matrix of shape (batch_size, num_users, num_users)

        Returns:
            torch.Tensor: Normalized precoding matrix
        """
        batch_size = precoding_matrix.shape[0]
        precoding_power = torch.sum(torch.abs(precoding_matrix)**2, dim=(1, 2))
        # precoding_power = torch.mean(torch.abs(precoding_matrix)**2, dim=2, keepdim=True)   # Normalized per transmitter
        scaling_factor = torch.sqrt(power_constraint / precoding_power).view(batch_size, 1, 1)
        return precoding_matrix * scaling_factor

    def generate_symbols(self, batch_size):
        """
        Generate random QAM symbols for each user

        Args:
            batch_size: Batch size

        Returns:
            torch.Tensor: Symbol vector of shape (batch_size, num_users)
        """
        # Randomly select symbols from constellation for each user in the batch
        indices = torch.randint(0, len(self.constellation_points), (batch_size, self.num_users))
        symbols = self.constellation_points[indices]
        return symbols, indices

    def compute_received_signals(self, precoding_matrix, symbols, channel_matrix, snr_db=20):
        """
        Compute received signals for all users

        Args:
            precoding_matrix: Precoding matrix of shape (batch_size, num_users, num_users)
            symbols: Symbol vector of shape (batch_size, num_users)
            channel_matrix: Channel matrix of shape (batch_size, num_users, num_users)
            snr_db: list of SNR in dB (batch_size)

        Returns:
            torch.Tensor: Received signals of shape (batch_size, num_users)
        """
        # Split into real and imaginary parts
        p_real, p_imag = precoding_matrix.real, precoding_matrix.imag
        s_real, s_imag = symbols.real.unsqueeze(2), symbols.imag.unsqueeze(2)
        h_real, h_imag = channel_matrix.real.to(torch.float32), channel_matrix.imag.to(torch.float32)

        # Complex multiplication between precoding_matrix and symbols using real operations
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        tx_real = torch.bmm(p_real, s_real) - torch.bmm(p_imag, s_imag)
        tx_imag = torch.bmm(p_real, s_imag) + torch.bmm(p_imag, s_real)

        # # Normalize Tx signals
        # tx_signal = torch.complex(tx_real, tx_imag)
        # scalar_factor = 1.0 / torch.mean(torch.abs(tx_signal)**2)
        # tx_signal = tx_signal * scalar_factor
        # tx_real, tx_imag = tx_signal.real, tx_signal.imag

        # Complex multiplication between channel_matrix and transmitted signal
        rx_real = torch.bmm(h_real, tx_real) - torch.bmm(h_imag, tx_imag)
        rx_imag = torch.bmm(h_real, tx_imag) + torch.bmm(h_imag, tx_real)
        rx_real, rx_imag = rx_real.squeeze(-1), rx_imag.squeeze(-1)

        # Add noise (AWGN)
        snr_linear = torch.tensor(10 ** (snr_db/10), dtype=torch.float32)
        noise_std = torch.sqrt(1 / (2 * snr_linear)).unsqueeze(-1)
        # noise_power = torch.abs(torch.mean(torch.abs(rx_real) ** 2 + torch.abs(rx_imag) ** 2)) / 10 ** (snr_db / 10)
        # noise_std = torch.tensor(torch.sqrt(torch.tensor(noise_power/2)))
        noise_real = torch.randn_like(rx_real) * noise_std
        noise_imag = torch.randn_like(rx_imag) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        # tx_snr = torch.mean(torch.abs(tx_signal)**2) / torch.mean(torch.abs(noise)**2)
        # print(f"Precoded snr (dB): {10*torch.log10(tx_snr)}")

        rx_real_noisy = rx_real + noise_real
        rx_imag_noisy = rx_imag + noise_imag

        # Combine real and imaginary parts
        received = torch.complex(rx_real_noisy, rx_imag_noisy)
        

        return received

    # def compute_power_regularization_loss(self, precoding_matrix, target_power=None):
    #     """
    #     Args:
    #         precoding_matrix: Complex precoding matrix (batch_size, num_users, num_users)
    #         power_target: Target power level (optional)
    #     """
    #     current_power = torch.mean(torch.abs(precoding_matrix) ** 2, dim=(1, 2))
    #     power_loss = torch.mean((current_power - target_power) ** 2)
    #     return power_loss

    def compute_constellation_distance_loss(self, received_signals, original_symbols):
        """
        Compute distance from ideal constellation points

        Args:
            received_signals: Received signals of shape (batch_size, num_users)
            original_symbols: Original symbol vector of shape (batch_size, num_users)

        Returns:
            torch.Tensor: Average distance from ideal constellation points
        """
        batch_size = received_signals.shape[0]
        # Ensure received symbols have same power distribution as original
        power_ratio = (torch.mean(torch.abs(original_symbols)**2) + + 1e-8) / (torch.mean(torch.abs(received_signals)**2) + 1e-8)
        normalized_received = received_signals * torch.sqrt(power_ratio)

        m = nn.Softplus(beta=10, threshold=0.0)
        if self.constellation_size == 4:
            # Compute distance from ideal constellation points
            rotated_org_symbols = rotate_symbols_to_x_axis(original_symbols, original_symbols)
            rotated_rec_symbols = rotate_symbols_to_x_axis(normalized_received, original_symbols)
            distances1 = point_to_line_distance(rotated_rec_symbols, rotated_org_symbols, 45)
            distances2 = point_to_line_distance(rotated_rec_symbols, rotated_org_symbols, -45)
            distances2 = -distances2    # Negative it as norm vector point to the increase direction

            # distances = ((rotated_rec_symbols - rotated_org_symbols).real * np.tan(math.pi/4)
            #           - torch.abs(rotated_rec_symbols.imag)) * np.cos(math.pi/4)

            # distance_loss = torch.mean(m(distances1) + m(distances2))
            distance_loss = (m(distances1) + m(distances2)).mean()

        elif self.constellation_size == 8:
            rotated_org_symbols = rotate_symbols_to_x_axis(original_symbols, original_symbols)
            rotated_rec_symbols = rotate_symbols_to_x_axis(normalized_received, original_symbols)
            distances1 = point_to_line_distance(rotated_rec_symbols, rotated_org_symbols)
            distances2 = point_to_line_distance(rotated_rec_symbols, rotated_org_symbols, -22.5)
            # Determine distance sign, if inside, negative it
            distances2 = -distances2

            # distance_loss = torch.mean(m(distances1) + m(distances2))
            distance_loss = (m(distances1) + m(distances2)).mean()
            # print(f"rotated rec symbol = {rotated_rec_symbols} and rotated org symbol = {rotated_org_symbols}")
            # print(f"AB = {AB} and AR = {AR}")
            # print(f"distance 1: {distances1}")
            # print(f"distance 2: {distances2}")

        elif self.constellation_size == 16:
            rotated_org_symbols = rotate_symbols_to_x_axis(original_symbols, original_symbols)
            rotated_rec_symbols = rotate_symbols_to_x_axis(normalized_received, original_symbols)
            distances1 = torch.zeros(batch_size, self.num_users)
            distances2 = torch.zeros(batch_size, self.num_users)
            for b in range(batch_size):
                for u in range(self.num_users):
                    if torch.abs(original_symbols[b,u].real) == 3 and torch.abs(original_symbols[b,u].imag) == 3:
                        # Corner point
                        distances1[b,u] = point_to_line_distance(rotated_rec_symbols[b, u], rotated_org_symbols[b, u])
                        distances2[b,u] = point_to_line_distance(rotated_rec_symbols[b, u], rotated_org_symbols[b, u], -45)
                        distances2 = -distances2    # Negative it as norm vector point to the increase direction
                    elif torch.abs(original_symbols.real[b,u]) == 3 or torch.abs(original_symbols[b,u].imag) == 3:
                        # Edge point
                        distances1[b, u] = - (rotated_rec_symbols[b, u].real - rotated_org_symbols[b, u].real) -\
                                      (1 - torch.abs(rotated_rec_symbols[b, u].imag))
                    else:
                        distances1[b, u] = torch.abs(rotated_rec_symbols[b, u] - rotated_org_symbols[b, u])

            # distance_loss = torch.mean(m(distances1) + m(distances2))
            distance_loss = (m(distances1) + m(distances2)).mean()


        return distance_loss
    

    # def compute_constellation_distance_loss(self, received_signals, original_symbols):
    #     """
    #     Compute distance from ideal constellation points

    #     Args:
    #         received_signals: Received signals of shape (batch_size, num_users)
    #         original_symbols: Original symbol vector of shape (batch_size, num_users)

    #     Returns:
    #         torch.Tensor: Average distance from ideal constellation points
    #     """
    #     original_power = torch.mean(torch.abs(original_symbols)**2, dim=-1, keepdim=True)
    #     original_norm = original_symbols / torch.sqrt(original_power + 1e-8)

    #     # Normalize received signals to match constellation scale
    #     # First compute average power of received signals per user
    #     avg_power = torch.mean(torch.abs(received_signals) ** 2, dim=0)

    #     # Scale factor to normalize to unit power (same as constellation points)
    #     scale_factor = 1.0 / torch.sqrt(avg_power + 1e-8)

    #     # Normalize received signals
    #     normalized_received = received_signals * scale_factor.unsqueeze(0)

    #     # Calculate Euclidean distance in complex plane
    #     distances = torch.abs(normalized_received - original_norm)

    #     # Square error (MSE-like loss)
    #     distance_loss = torch.mean(distances ** 2)

    #     return distance_loss

    def compute_symbol_error_rate(self, received_signals, original_symbols):
        """
        Compute symbol error rate

        Args:
            received_signals: Received signals of shape (batch_size, num_users)
            original_symbols: Original symbol vector of shape (batch_size, num_users)

        Returns:
            torch.Tensor: Symbol error rate
        """
        batch_size = received_signals.shape[0]

        # Ensure received symbols have same power distribution as original
        power_ratio = torch.mean(torch.abs(original_symbols)**2) / torch.mean(torch.abs(received_signals)**2)
        normalized_received = received_signals * torch.sqrt(power_ratio)

        # For each received symbol, find nearest constellation point
        detected_symbols = torch.zeros_like(normalized_received)

        for b in range(batch_size):
            for u in range(self.num_users):
                received = normalized_received[b, u]

                # Calculate distance to each constellation point
                distances = torch.abs(self.constellation_points - received)

                # Find index of closest constellation point
                closest_idx = torch.argmin(distances)

                # Map to corresponding constellation point
                detected_symbols[b, u] = self.constellation_points[closest_idx]

        # Count symbol errors
        errors = (detected_symbols != original_symbols).float()
        ser = torch.mean(errors)

        return ser

def generate_AoD(num_samples, num_users, positions, f_MHz=2400):
    distances = np.zeros((num_users, num_users))
    azimuth_angles = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(num_users):
            if i != j:
                dx = positions[i,0] - positions[j,0]
                dy = positions[i,1] - positions[j,1]
                distances[i][j] = np.sqrt(dx**2 + dy**2)
                azimuth_angles[i, j] = np.arctan2(dy, dx) * 180 / np.pi
    # Calculate antenna propogation phase
    # Total propagation phase (2π * distance / wavelength)
    wavelength = 3e8 / (f_MHz * 1e6)
    propagation_phase = 2 * np.pi * distances / wavelength
    # Polarization mismatch angle
    geo_phase = np.random.uniform(-5, 5, size=(num_users, num_users)) * np.pi / 180
    phase = propagation_phase + geo_phase

    return phase, distances

def generate_channel_data(num_samples, num_users, k_factor_dB=0, mode='rician'):
    """
    Generate channel coefficient for rician channel
    Args:
        num_samples: number of batch size
        num_users: number of users
        phase: phase of the channel
        path_loss_dB: received path loss in dB
        k_factor_dB: control contribution for LOS annd NLOS k = 0, LOS 1/2, NLOS 1/2
    Returns:
        torch.Tensor: Normalized channel matrices of shape (num_samples, num_users, num_users)
        torch.Tensor: Channel norms of shape (num_samples, 1, 1)
    """
    # Parameters
    f_MHz = 2400  # in MHz
    positions = np.array([
        [0, 0],
        [400, 150],
        [120, 400]
        ])
    alpha = 2.5
    G_T = 20   # antenna gain
    phase, distances = generate_AoD(num_samples, num_users, positions, f_MHz=f_MHz)

    d_km = distances / 1000
    d_km[d_km==0] = 0.4
    path_loss_dB = 32.44 + 10 * alpha * np.log10(f_MHz) + 10 * alpha * np.log10(d_km) - 2*G_T
    path_loss_linear = 10 ** (path_loss_dB / 10)
    path_loss_factor = torch.tensor(1 / np.sqrt(path_loss_linear))

    if mode == 'rician':
        K = 10 ** (k_factor_dB / 10)  # Convert K from dB to linear
        # LOS component with deterministic phase
        los_mag = torch.sqrt(torch.tensor(K / (K + 1)))
        los_component = los_mag * np.exp(1j * phase)
        # NLOS component
        nlos_mag = torch.sqrt(torch.tensor(1 / (K + 1)))

        # Generate complex Gaussian random variables (X+jY)
        X = torch.normal(0.0, np.sqrt(1/2), size=(num_samples, num_users, num_users))
        Y = torch.normal(0.0, np.sqrt(1/2), size=(num_samples, num_users, num_users))
        scatter_component = (X + 1j * Y)

        # Combine LoS and scattered components
        fading = los_component + nlos_mag * scatter_component
        h = path_loss_factor * fading
        h = h.to(torch.complex64)
    else:
        raise ValueError(f"Unknown channel mode: {mode}")

    h_power = torch.mean(torch.abs(h) ** 2, dim=(1, 2), keepdim=True)
    return h / torch.sqrt(h_power + 1e-8), torch.sqrt(h_power)

# def generate_channel_data(num_samples, num_users, mode='rayleigh'):
#     """
#     Generate channel state information data

#     Args:
#         num_samples: Number of samples
#         num_users: Number of users
#         mode: Channel model ('rayleigh' or '
        
#         ')

#     Returns:
#         torch.Tensor: Channel matrices of shape (num_samples, num_users, num_users)
#     """
#     # Parameters
#     f_MHz = 2400  # in MHz
#     d_km = 0.05   # in kilo-meters
#     alpha = 2.5
#     G_T = 20   # antenna gain
#     path_loss_dB = 32.44 + 10 * alpha * np.log10(f_MHz) + 10 * alpha * np.log10(d_km) - 2*G_T

#     if mode == 'rayleigh':
#         # Rayleigh fading channel
#         h_real = torch.randn(num_samples, num_users, num_users) / np.sqrt(2)
#         h_imag = torch.randn(num_samples, num_users, num_users) / np.sqrt(2)
#         h = torch.complex(h_real, h_imag)
#     elif mode == 'rician':
#          # Convert path loss from dB to linear
#         path_loss_linear = 10 ** (path_loss_dB / 10)
#         path_loss_factor = 1 / np.sqrt(path_loss_linear)
#         # Rician fading channel with K-factor = 1
#         k_factor_dB = 0.0  # Weak LOS 1/2 LOS, 1/2 NLOS
#         K = 10 ** (k_factor_dB / 10)  # Convert K from dB to linear
#         # LOS component
#         los_mag = torch.sqrt(torch.tensor(K / (K + 1)))
#         # NLOS component
#         nlos_mag = torch.sqrt(torch.tensor(1 / (K + 1)))

#         # Generate complex Gaussian random variables (X+jY)
#         X = torch.normal(0.0, 0.5, size=(num_samples, num_users, num_users))
#         Y = torch.normal(0.0, 0.5, size=(num_samples, num_users, num_users))
#         scatter_component = X + 1j * Y

#         # Combine LoS and scattered components
#         fading = los_mag + nlos_mag * scatter_component
#         h = path_loss_factor * fading
#     elif mode == 'path_loss':
#         path_loss_linear = 10 ** (path_loss_dB / 10)
#         path_loss_factor = 1 / np.sqrt(path_loss_linear)
#         phase = torch.distributions.Uniform(0, np.pi).sample((num_samples, num_users, num_users))   # limited phase from 0 to pi
#         h = path_loss_factor * np.exp(1j * phase)
#     else:
#         raise ValueError(f"Unknown channel mode: {mode}")

#     h_power = torch.mean(torch.abs(h) ** 2, dim=(1, 2), keepdim=True) 
#     return h / torch.sqrt(h_power + 1e-8), torch.sqrt(h_power)

def train_symbol_level_precoding_network(snr_db_range=(0.0, 31.0)):
    # Parameters
    num_users = 3
    constellation_size = 16
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    k_factor_dB = 10

    # Generate training data
    num_train_samples = 10000
    channel_matrices, channel_norm = generate_channel_data(num_train_samples, num_users, k_factor_dB, mode='rician')

    # Create dataset and dataloader
    train_dataset = TensorDataset(channel_matrices, channel_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = SymbolLevelPrecodingNetwork(num_users=num_users, constellation_size=constellation_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    loss_history = []
    ser_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ser = 0.0
        num_batches = 0

        for channel_batch, channel_norm_batch in train_loader:
            channels = channel_batch
            batch_size = channels.shape[0]

            # Pick SNR smoothly from low to high as epoch increases cause rician 
            # snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
            # snr_per_user = torch.tensor(snr_db).repeat(batch_size, num_users)
            snr_step = 1.0
            all_snr_values = np.arange(snr_db_range[0], snr_db_range[1] + snr_step, snr_step)
            snr_per_sample = np.random.choice(all_snr_values, size=batch_size)
            snr_per_user = torch.tensor(snr_per_sample).unsqueeze(1).repeat(1, num_users).float()
            snr_db = np.mean(snr_per_sample)  # For logging

            # Generate random symbols for each user
            symbols, _ = model.generate_symbols(batch_size)

            # Zero gradients
            optimizer.zero_grad()


            # Forward pass: generate symbol-level precoding matrix
            precoding_matrix = model(channels, symbols, snr_per_user)

            # Compute received signals
            # received_signals not normal
            received_signals = model.compute_received_signals(
                precoding_matrix, symbols, channels, snr_db=snr_per_sample
            )

            # Compute constellation distance loss
            loss = model.compute_constellation_distance_loss(received_signals, symbols) 
                    # 0.1 * model.compute_power_regularization_loss(precoding_matrix, power_target)

            # Compute symbol error rate (for monitoring)
            # ser = model.compute_CI_ser(received_signals, symbols)
            ser = model.compute_symbol_error_rate(received_signals, symbols)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Record metrics
            epoch_loss += loss.item()
            epoch_ser += ser.item()
            num_batches += 1

        # Step the learning rate scheduler
        scheduler.step()

        # Average metrics for the epoch
        avg_loss = epoch_loss / num_batches
        avg_ser = epoch_ser / num_batches

        loss_history.append(avg_loss)
        ser_history.append(avg_ser)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, SER: {avg_ser:.5f}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Constellation Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(ser_history)
    plt.title('Symbol Error Rate')
    plt.xlabel('Epoch')
    plt.ylabel('SER')

    plt.tight_layout()
    plt.show()

    return model, loss_history, ser_history

def test_symbol_level_precoding(model, snr_db_list=None):
    """
    Test the symbol-level precoding model at different SNR levels
    """
    if snr_db_list is None:
        snr_db_list = list(range(0, 31, 2))  
        # snr_db_list = np.arange(0,30,3)

    num_users = model.num_users
    num_test_samples = 1000
    k_factor_dB = 10

    # Generate test data
    test_channels, channel_norm = generate_channel_data(num_test_samples, num_users, k_factor_dB, mode='rician')

    # Results storage
    sl_ser_results = []
    zf_ser_results = []
    uncoded_ser_results = []
    rzf_ser_results = []
    loss_history = []

    # Test at different SNR levels
    for snr_db in snr_db_list:
        noise_power = 10 ** (-snr_db / 10)
        # Test symbol-level precoding
        model.eval()
        sl_symbol_errors = 0
        sl_total_symbols = 0
        epoch_loss = 0
        batch_count = 0

        with torch.no_grad():
            for i in range(0, num_test_samples, 32):
                # Get batch
                end_idx = min(i + 32, num_test_samples)
                batch_size = end_idx - i
                channels_batch = test_channels[i:end_idx]
                channel_norm_batch = channel_norm[i:end_idx]

                # Generate random symbols
                symbols, _ = model.generate_symbols(batch_size)

                # Generate symbol-level precoding matrix
                snr_per_user = torch.tensor(snr_db, dtype=torch.float32).repeat(batch_size, num_users)
                precoding_matrix = model(channels_batch, symbols, snr_per_user)

                # Compute received signals
                snr_array = np.full(batch_size, snr_db)
                received_signals = model.compute_received_signals(
                    precoding_matrix, symbols, channels_batch, snr_db=snr_array
                )
                # Compute constellation distance loss
                loss = model.compute_constellation_distance_loss(received_signals, symbols)
                    # 0.1 * model.compute_power_regularization_loss(precoding_matrix, target_power)

                # Compute SER
                # ser = model.compute_CI_ser(received_signals, symbols)
                ser = model.compute_symbol_error_rate(received_signals, symbols)

                sl_symbol_errors += ser.item() * batch_size * num_users
                sl_total_symbols += batch_size * num_users

                epoch_loss += loss.item()

        sl_ser = sl_symbol_errors / sl_total_symbols
        sl_ser_results.append(sl_ser)

        # Test Zero-Forcing precoding
        zf_symbol_errors = 0
        zf_total_symbols = 0

        # Loss function
        avg_loss = epoch_loss / (i+1)
        loss_history.append(avg_loss)

        for i in range(0, num_test_samples, 32):
            # Get batch
            end_idx = min(i + 32, num_test_samples)
            batch_size = end_idx - i
            channels_batch = test_channels[i:end_idx]

            # Generate random symbols
            symbols, original_indices = model.generate_symbols(batch_size)

            # ZF precoding matrices
            zf_precoding_matrices = []

            for b in range(batch_size):
                H = channels_batch[b]
                # Pseudo-inverse for ZF precoding hamilton 
                H_H = H.conj().transpose(0, 1)
                try:
                    P_zf = H_H @ torch.inverse(H @ H_H)

                    # Normalize power
                    norm = torch.sqrt(torch.sum(torch.abs(P_zf) ** 2, dim=0, keepdim=True))
                    P_zf_norm = P_zf / (norm + 1e-8)

                    zf_precoding_matrices.append(P_zf_norm)
                except:
                    # Skip if matrix inversion fails
                    continue

            if not zf_precoding_matrices:
                continue

            # Stack matrices
            zf_precoding = torch.stack(zf_precoding_matrices)
            valid_batch_size = zf_precoding.shape[0]
            valid_symbols = symbols[:valid_batch_size]
            valid_channels = channels_batch[:valid_batch_size]

            # Compute received signals with ZF precoding
            received_zf = []
            zf_precoding = zf_precoding.to(torch.complex64)
            valid_symbols = valid_symbols.to(torch.complex64)

            for b in range(valid_batch_size):
                # Transmitted signal
                tx = zf_precoding[b] @ valid_symbols[b]

                # Channel effect
                rx = valid_channels[b] @ tx

                # Add noise
                if torch.is_complex(rx):
                    # Generate real-valued noise tensors with the same shape as rx
                    noise_real = torch.randn(rx.shape, device=rx.device) * torch.sqrt(torch.tensor(noise_power/2))
                    noise_imag = torch.randn(rx.shape, device=rx.device) * torch.sqrt(torch.tensor(noise_power/2))

                    # Create complex noise
                    noise = torch.complex(noise_real, noise_imag)
                else:
                    # If rx is not complex, handle accordingly
                    noise = torch.randn_like(rx) * torch.sqrt(torch.tensor(noise_power))

                rx_noisy = rx + noise

                received_zf.append(rx_noisy)

            if received_zf:
                received_zf = torch.stack(received_zf)

                # Compute SER for ZF
                # Normalize received signals
                avg_power = torch.mean(torch.abs(received_zf) ** 2, dim=1, keepdim=True)
                scale_factor = 1.0 / torch.sqrt(avg_power + 1e-8)
                normalized_received = received_zf * scale_factor

                # For each received symbol, find nearest constellation point
                errors = 0
                for b in range(valid_batch_size):
                    for u in range(num_users):
                        received = normalized_received[b, u]

                        # Find closest constellation point
                        distances = torch.abs(model.constellation_points - received)
                        closest_idx = torch.argmin(distances)
                        detected = model.constellation_points[closest_idx]

                        # Check for error
                        if detected != valid_symbols[b, u]:
                            errors += 1

                zf_symbol_errors += errors
                zf_total_symbols += valid_batch_size * num_users

        if zf_total_symbols > 0:
            zf_ser = zf_symbol_errors / zf_total_symbols
            zf_ser_results.append(zf_ser)
        else:
            zf_ser_results.append(1.0)  # Default to worst case if all inversions failed
            
        # Test Regularized Zero-forcing (ZF) precoding
        rzf_symbol_errors = 0
        rzf_total_symbols = 0
        alpha = num_users / (10 ** (snr_db / 10))  # Regularization parameter

        for i in range(0, num_test_samples, 32):
            end_idx = min(i + 32, num_test_samples)
            batch_size = end_idx - i
            channels_batch = test_channels[i:end_idx]
            symbols, original_indices = model.generate_symbols(batch_size)

            rzf_precoding_matrices = []

            for b in range(batch_size):
                H = channels_batch[b]
                H_H = H.conj().transpose(0, 1)
                try:
                    # RZF formula: H^H (H H^H + alpha*I)^-1
                    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
                    P_rzf = H_H @ torch.inverse(H @ H_H + alpha * eye)

                    # Normalize power
                    norm = torch.sqrt(torch.sum(torch.abs(P_rzf) ** 2, dim=0, keepdim=True))
                    P_rzf_norm = P_rzf / (norm + 1e-8)

                    rzf_precoding_matrices.append(P_rzf_norm)
                except:
                    continue

            if not rzf_precoding_matrices:
                continue

            rzf_precoding = torch.stack(rzf_precoding_matrices)
            valid_batch_size = rzf_precoding.shape[0]
            valid_symbols = symbols[:valid_batch_size]
            valid_channels = channels_batch[:valid_batch_size]
            rzf_precoding = rzf_precoding.to(torch.complex64)
            valid_symbols = valid_symbols.to(torch.complex64)

            received_rzf = []
            for b in range(valid_batch_size):
                tx = rzf_precoding[b] @ valid_symbols[b]
                rx = valid_channels[b] @ tx

                # Add noise
                if torch.is_complex(rx):
                    noise_real = torch.randn(rx.shape, device=rx.device) * torch.sqrt(torch.tensor(noise_power/2))
                    noise_imag = torch.randn(rx.shape, device=rx.device) * torch.sqrt(torch.tensor(noise_power/2))
                    noise = torch.complex(noise_real, noise_imag)
                else:
                    noise = torch.randn_like(rx) * torch.sqrt(torch.tensor(noise_power))
                rx_noisy = rx + noise
                received_rzf.append(rx_noisy)

            if received_rzf:
                received_rzf = torch.stack(received_rzf)
                avg_power = torch.mean(torch.abs(received_rzf) ** 2, dim=1, keepdim=True)
                scale_factor = 1.0 / torch.sqrt(avg_power + 1e-8)
                normalized_received = received_rzf * scale_factor

                errors = 0
                for b in range(valid_batch_size):
                    for u in range(num_users):
                        received = normalized_received[b, u]
                        distances = torch.abs(model.constellation_points - received)
                        closest_idx = torch.argmin(distances)
                        detected = model.constellation_points[closest_idx]
                        if detected != valid_symbols[b, u]:
                            errors += 1

                rzf_symbol_errors += errors
                rzf_total_symbols += valid_batch_size * num_users

        if rzf_total_symbols > 0:
            rzf_ser = rzf_symbol_errors / rzf_total_symbols
            rzf_ser_results.append(rzf_ser)
        else:
            rzf_ser_results.append(1.0)

        print(f"SNR: {snr_db} dB, Symbol-Level SER: {sl_ser:.4e}, ZF SER: {zf_ser_results[-1]:.4e}, RZF SER: {rzf_ser_results[-1]:.4e}")
        
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_list, sl_ser_results, 'x-', label='SLP')
    plt.semilogy(snr_db_list, zf_ser_results, 'v--', label='ZF')
    plt.semilogy(snr_db_list, rzf_ser_results, '^--', label='RZF')
    # plt.semilogy(snr_db_list, uncoded_ser_results, 'o-', label='Uncoded')
    plt.title('Symbol Error Rate vs. SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sl_ser_results, zf_ser_results, loss_history

def visualize_constellation(model, snr_db=10):
    """
    Visualize received constellation points for symbol-level precoding
    """
    num_users = model.num_users
    num_samples = 500
    noise_power = 10 ** (-snr_db / 10)

    # Generate channel matrices
    channels, _ = generate_channel_data(num_samples, num_users, mode='rician')

    # Generate random symbols
    symbols, _ = model.generate_symbols(num_samples)

    snr_per_user = torch.tensor(snr_db).repeat(num_samples, num_users)

    # Generate symbol-level precoding matrices
    model.eval()
    with torch.no_grad():
        precoding_matrices = model(channels, symbols, snr_per_user)

    # Compute received signals
    received_signals = model.compute_received_signals(
        precoding_matrices, symbols, channels, snr_db=snr_db
    )

    # Normalize received signals
    avg_power = torch.mean(torch.abs(received_signals) ** 2, dim=0)
    scale_factor = 1.0 / torch.sqrt(avg_power + 1e-8)
    normalized_received = received_signals * scale_factor.unsqueeze(0)

    # Plot constellation for each user
    plt.figure(figsize=(16, 12))

    for u in range(min(num_users, 4)):  # Plot up to 4 users
        plt.subplot(2, 2, u+1)

        # Original symbols for this user
        user_symbols = symbols[:, u].cpu().numpy()

        # Received signals for this user
        user_received = normalized_received[:, u].cpu().numpy()

        # Plot original constellation points
        for i, point in enumerate(model.constellation_points):
            point_np = complex(point.item())
            plt.scatter(point_np.real, point_np.imag, s=100, c='red', marker='x', label=f'QAM Point {i+1}' if u == 0 else "")

        # Plot received signals
        plt.scatter(user_received.real, user_received.imag, s=10, alpha=0.5, c='blue', label='Received' if u == 0 else "")

        plt.title(f'User {u+1} Constellation, SNR={snr_db} dB')
        plt.grid(True)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        # if u == 0:
        #     plt.legend()

    plt.tight_layout()
    plt.show()

def rotate_symbols_to_x_axis(received_symbols, original_qam_symbols):
    """
    Rotate received symbols to the positive x-axis based on the angle of original 8QAM/8PSK symbols.
    
    Args:
        received_symbols: Complex tensor of shape (batch_size, num_users)
        original_qam_symbols: Complex tensor of shape (batch_size, num_users)
    
    Returns:
        Rotated symbols: Complex tensor of shape (batch_size, num_users)
    """
    # Calculate the angle of each original symbol (in radians)
    original_angles = torch.angle(original_qam_symbols)
    
    # Create rotation factors to rotate to the positive x-axis
    # To rotate to 0 degrees, we need to rotate by the negative of the original angle
    # Use e^(-j*θ) for rotation, where θ is the original angle
    rotation_factors = torch.exp(-1j * original_angles)
    
    # Apply the rotation to the received symbols
    rotated_symbols = received_symbols * rotation_factors
    
    return rotated_symbols

def point_to_line_distance(point, line_start, angle_degrees=45):
    """
    point: (x, y)
    line_start: (x1, y1)  point on the line
    """
    # Extract coordinates
    x0, y0 = point.real, point.imag
    x1, y1 = line_start.real, line_start.imag

    # Convert angle to radians
    angle_rad = torch.tensor(np.radians(angle_degrees))

    # Calculate distance
    a = -torch.sin(angle_rad)
    b = torch.cos(angle_rad)
    c = -a * x1 - b * y1

    distances = (a * x0 + b * y0 + c) / torch.sqrt(a**2 + b**2)     # the sign of results reveal the relative locati
    # print(f"distances: {distances}")

    return distances

# Example use
if __name__ == "__main__":
    # Train the model
    model, loss_history, ser_history = train_symbol_level_precoding_network()
    torch.save(model.state_dict(), './saved_models/model_weights3_4_3user_16QAM.pth')
    print("Model saved successfully!")

    # Test at different SNR levels
    sl_ser, zf_ser, loss_hist = test_symbol_level_precoding(model)

    # Compare Loss
    # Plot training history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Constellation Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(loss_hist)
    plt.title('Testing constellation distance loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

    # Visualize constellation
    visualize_constellation(model, snr_db=15)

    # # print(model)