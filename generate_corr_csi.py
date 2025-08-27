import numpy as np
import torch
from scipy.spatial.distance import cdist
import sys

def generate_channel_data(num_time_steps, tx_init, rx_init,
                          velocities=50, vibration_std=10, area_size=200, k_factor_dB=10, 
                          mode='rician', rho_spatial=0.01, rho_temporal=0.98, mismatch_deg=None):
    """
    Generate channel data with spatial and temporal correlation
    
    Args:
        num_time_steps: Number of time steps
        tx_init: Initial transmitter positions (num_users, 2)
        rx_init: Initial receiver positions (num_users, 2)
        velocities: Movement velocity
        vibration_std: Standard deviation of random vibrations
        area_size: Control the slope of correlation decay vs distance
        k_factor_dB: Rician K-factor in dB
        mode: Channel mode ('rician')
        rho_spatial: Spatial correlation coefficient (0-1)
        rho_temporal: Temporal correlation coefficient (0-1)
    """
    num_users = rx_init.shape[0]

    def compute_path_loss(tx_positions, rx_positions, mismatch_deg=None):
        # Compute path loss between all tx-rx pairs
        """
        mismatch_deg: antenna mismatch in degree
        """
        f_MHz = 2400  # in MHz
        alpha = 2.5
        G_T = 20   # antenna gain
        azimuth_angles = np.zeros((num_users, num_users))
        distances = np.zeros((num_users, num_users))
        
        for i in range(num_users):
            for j in range(num_users):
                dx = tx_positions[i,0] - rx_positions[j,0]
                dy = tx_positions[i,1] - rx_positions[j,1]
                azimuth_angles[i, j] = np.arctan2(dy, dx)
                distances[i][j] = np.sqrt(dx**2 + dy**2)

        d_km = distances / 1000
        path_loss_dB = 32.44 + 10 * alpha * np.log10(f_MHz) + 10 * alpha * np.log10(d_km) - 2*G_T
        path_loss_dB[path_loss_dB == -np.inf] = 0
        path_loss_linear = 10 ** (path_loss_dB / 10)
        path_loss_factor = torch.tensor(1 / np.sqrt(path_loss_linear))

        # Phase shift
        wavelength = 3e8 / (f_MHz * 1e6)
        propagation_phase = 2 * np.pi * distances / wavelength
        if mismatch_deg is None:
            # Training: no mismatch
            # Polarization mismatch angle
            phase = propagation_phase
        else:
            # Testing: controlled mismatch, constant antenna mismatch
            np.random.seed(36)
            geo_phase = np.full((num_users, num_users), mismatch_deg * np.pi / 180)      # +mismatch
            # geo_phase = np.random.uniform(-5, 5, size=(num_users, num_users)) * np.pi / 180
            phase = propagation_phase + geo_phase

        return path_loss_factor, phase

    def compute_spatial_correlation(positions, rho):
        """Compute spatial correlation matrix based on distances"""
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        R = rho ** (dists / area_size)  # Normalize by area_size
        return R + 1e-6 * np.eye(len(positions))  # Add small diagonal term for stability

    def apply_spatial_correlation(R_tx, R_rx, uncorr_noise):
        """
        Apply spatial correlation to uncorrelated noise using Kronecker model
        """
        # Cholesky decomposition for stable correlation application
        try:
            L_tx = np.linalg.cholesky(R_tx)
            L_rx = np.linalg.cholesky(R_rx)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if Cholesky fails
            eigenvals_tx, eigenvecs_tx = np.linalg.eigh(R_tx)
            eigenvals_tx = np.maximum(eigenvals_tx, 1e-8)  # Ensure positive
            L_tx = eigenvecs_tx @ np.diag(np.sqrt(eigenvals_tx))
            eigenvals_rx, eigenvecs_rx = np.linalg.eigh(R_rx)
            eigenvals_rx = np.maximum(eigenvals_rx, 1e-8)  # Ensure positive
            L_rx = eigenvecs_rx @ np.diag(np.sqrt(eigenvals_rx))
        
        # Apply correlation: H_corr = L_rx @ H_uncorr @ L_tx^T
        corr_noise = L_rx @ uncorr_noise @ L_tx.T
        original_power = np.mean(np.abs(uncorr_noise)**2)
        corr_power = np.mean(np.abs(corr_noise)**2)
        normalization_factor = np.sqrt(original_power / corr_power)
        return corr_noise * normalization_factor

    def generate_uav_position():
        # Initialize positions
        tx_positions = np.zeros((num_time_steps, num_users, 2))
        rx_positions = np.zeros((num_time_steps, num_users, 2))
        v_std = np.zeros((num_time_steps, num_users, 2))

        # Time evolution
        for t in range(num_time_steps):
            if t == 0:
                tx_positions[t] = tx_init
                rx_positions[t] = rx_init
            else:
                # Update positions with directed movement and random vibrations
                tx_positions[t] = (tx_positions[t-1] + velocities +
                                  np.random.normal(0, vibration_std, (num_users, 2)))
                rx_positions[t] = (rx_positions[t-1] + velocities +
                                  np.random.normal(0, vibration_std, (num_users, 2)))

                # Wrap around boundaries
                tx_positions[t] = tx_positions[t] % area_size
                rx_positions[t] = rx_positions[t] % area_size

        return tx_positions, rx_positions

    def generate_csi_with_correlation(path_loss_factor, k_factor_dB, phase, 
                                    R_tx, R_rx, prev_fading=None, pre_los=None, mode='rician'):
        """
        Generate CSI with spatial and temporal correlation
        """
        if mode == 'rician':
            K = 10 ** (k_factor_dB / 10)  # Convert K from dB to linear
            
            # LOS component with deterministic phase
            los_mag = np.sqrt(K / (K + 1))
            los_component = los_mag * np.exp(1j * phase)
            
            # NLOS component magnitude
            nlos_mag = np.sqrt(1 / (K + 1))
            
            # Generate uncorrelated complex Gaussian noise
            np.random.seed(36)  # Use different seed for each time step
            X = np.random.normal(0, np.sqrt(1/2), size=(num_users, num_users))
            Y = np.random.normal(0, np.sqrt(1/2), size=(num_users, num_users))
            uncorr_scatter = X + 1j * Y
            
            # Apply spatial correlation to the scattered component
            corr_scatter = apply_spatial_correlation(R_tx, R_rx, uncorr_scatter)
            
            # Apply temporal correlation if previous fading exists
            if prev_fading is not None:
                # Extract scattered component from previous fading
                prev_scatter = (prev_fading - pre_los) / nlos_mag
                # Apply temporal correlation: h[n] = ρ*h[n-1] + √(1-ρ²)*w[n]
                temporal_factor = np.sqrt(1 - rho_temporal**2)
                corr_scatter = rho_temporal * prev_scatter + temporal_factor * corr_scatter
                # corr_scatter = corr_scatter
                # print(f"corr_scatter: {corr_scatter}")
            
            # Combine LoS and scattered components
            fading = los_component + nlos_mag * corr_scatter
            
            # Convert path_loss_factor to numpy if it's a tensor
            if hasattr(path_loss_factor, 'numpy'):
                path_loss_np = path_loss_factor.numpy()
            else:
                path_loss_np = path_loss_factor
                
            h = path_loss_np * fading
            
        else:
            raise ValueError(f"Unknown channel mode: {mode}")

        h_power = np.mean(np.abs(h) ** 2)
        normalized_h = h / np.sqrt(h_power + 1e-8)
        # normal = np.random.normal(0,1,size=(num_users, num_users))
        # normalized_h += normal
        return normalized_h, np.sqrt(h_power), fading, los_component

    # Generate UAV positions
    tx_positions, rx_positions = generate_uav_position()
    
    # Initialize output arrays
    H = np.zeros((num_time_steps, num_users, num_users), dtype=np.complex64)
    h_norm = np.zeros((num_time_steps, num_users, num_users), dtype=np.float32)
    
    # Store previous fading for temporal correlation
    prev_fading = None
    prev_los = None
    
    distances = np.zeros((num_time_steps, num_users, num_users))
    r_tx_all = np.zeros((num_time_steps, num_users, num_users), dtype=np.float32)
    r_rx_all = np.zeros((num_time_steps, num_users, num_users), dtype=np.float32)
    # Generate channel matrices for each time step
    for t in range(num_time_steps):
        # Compute path loss and phase for current positions
        path_loss_factor, phase = compute_path_loss(tx_positions[t], rx_positions[t], mismatch_deg=None)
        
        # Compute spatial correlation matrices based on current positions
        R_tx = compute_spatial_correlation(tx_positions[t], rho_spatial)
        R_rx = compute_spatial_correlation(rx_positions[t], rho_spatial)
        r_tx_all[t,:,:] = R_tx
        r_rx_all[t,:,:] = R_rx
        
        # Generate CSI with spatial and temporal correlation
        H[t,:,:], h_norm[t,:,:], current_fading, current_los = generate_csi_with_correlation(
            path_loss_factor, k_factor_dB, phase, R_tx, R_rx, prev_fading, prev_los, mode=mode)
        
        # Store current fading for next time step
        prev_fading = current_fading
        prev_los = current_los

        # Calculate distances
        distances[t,:,:] = cdist(tx_positions[t], rx_positions[t], metric='euclidean')
    link_map = create_link_map(tx_positions, rx_positions, beamwidth_degrees=30, main_gain_db=20, side_lobe_db=0)
    # H = H * link_map
    if not torch.is_tensor(H):
        H = torch.tensor(H)

    # Add white noise
    noise_real = torch.randn(num_time_steps, num_users, num_users) * torch.sqrt(torch.tensor(1 / 2))
    noise_imag = torch.randn(num_time_steps, num_users, num_users) * torch.sqrt(torch.tensor(1 / 2))
    noise = torch.complex(noise_real, noise_imag)
    H_noisy = H + noise

    return H_noisy, torch.tensor(distances).float(), torch.tensor(r_tx_all), torch.tensor(r_rx_all)

def antenna_gain_pattern_simple(angle_deg, beamwidth_deg, main_gain_db=0, side_lobe_db=-20):
    """
    Simple antenna gain pattern.
    
    Parameters:
    angle_deg: angle from main beam direction in degrees
    beamwidth_deg: 3dB beamwidth in degrees
    main_gain_db: main lobe gain in dB
    side_lobe_db: side lobe gain in dB
    
    Returns:
    gain in linear scale (not dB)
    """
    half_beamwidth = beamwidth_deg / 2
    
    if angle_deg <= half_beamwidth:
        # Main lobe: cosine taper from peak to 3dB down at edge
        normalized_angle = angle_deg / half_beamwidth  # 0 to 1
        gain_db = main_gain_db - 3 * (normalized_angle ** 2)  # 3dB down at edge
    else:
        # Side lobe: constant low gain
        gain_db = side_lobe_db
    # Convert dB to linear scale
    return 10 ** (gain_db / 10)

def antenna_gain_pattern_realistic(angle_deg, beamwidth_deg, main_gain_db=0, 
                                 first_null_db=-40, side_lobe_db=-20):
    """
    More realistic antenna pattern with nulls and varying side lobes.
    """
    half_beamwidth = beamwidth_deg / 2
    
    if angle_deg <= half_beamwidth:
        # Main lobe: smooth rolloff
        normalized_angle = angle_deg / half_beamwidth
        gain_db = main_gain_db * np.cos(np.pi * normalized_angle / 2) ** 2
        gain_db = 10 * np.log10(gain_db) if gain_db > 0 else -60
    elif angle_deg <= 2 * half_beamwidth:
        # Transition region with first null
        gain_db = first_null_db
    else:
        # Side lobes with some variation
        # Add some oscillation to simulate multiple side lobes
        oscillation = 5 * np.sin(angle_deg * np.pi / (2 * half_beamwidth))
        gain_db = side_lobe_db + oscillation
    
    return 10 ** (gain_db / 10)

def create_link_map(tx_positions, rx_positions, beamwidth_degrees, pattern_type='simple',
                   main_gain_db=0, side_lobe_db=-20):
    """
    Create a link map for directional antennas pointing from tx_i to rx_i.
    
    Parameters:
    tx_positions: array of shape (T, N, 2) - transmitter positions over time
    rx_positions: array of shape (T, N, 2) - receiver positions over time  
    beamwidth_degrees: float - antenna beamwidth in degrees
    pattern_type: 'binary', 'simple' or' realistic
    
    Returns:
    link_map: array of shape (T, N, N) - 1 if rx_j is in tx_i's beam, 0 otherwise
    """
    T, num_users, _ = tx_positions.shape
    beamwidth_rad = np.radians(beamwidth_degrees)
    half_beamwidth = beamwidth_rad / 2
    
    # Initialize link map
    link_map = np.zeros((T, num_users, num_users), dtype=np.float32)
    
    for t in range(T):
        for i in range(num_users):  # For each transmitter tx_i
            # Get tx_i position and rx_i position (target direction)
            tx_pos = tx_positions[t, i]
            rx_target_pos = rx_positions[t, i]
            
            # Calculate main beam direction (from tx_i to rx_i)
            beam_vector = rx_target_pos - tx_pos
            beam_distance = np.linalg.norm(beam_vector)
            
            # Skip if tx and rx are at same position
            if beam_distance == 0:
                continue
                
            # Normalize beam direction
            beam_direction = beam_vector / beam_distance
            
            # Check all receivers to see if they're in the beam
            for j in range(num_users):
                if i == j:  # tx_i and rx_i must be connected
                    link_map[t, i, j] = 1
                    continue
                    
                # Vector from tx_i to rx_j
                rx_j_pos = rx_positions[t, j]
                to_rx_j = rx_j_pos - tx_pos
                distance_to_rx_j = np.linalg.norm(to_rx_j)
                
                # Skip if at same position
                if distance_to_rx_j == 0:
                    warnings.warn("Tx and Rx are collided")
                
                # Normalize direction to rx_j
                direction_to_rx_j = to_rx_j / distance_to_rx_j
                
                # Calculate angle between beam direction and direction to rx_j
                cos_angle = np.clip(np.dot(beam_direction, direction_to_rx_j), -1, 1)
                angle_deg = np.arccos(cos_angle) * 180 / np.pi
                
                if pattern_type == 'binary':
                    # Original binary approach
                    if angle_deg <= half_beamwidth:
                        link_map[t, i, j] = 1.0
                    else:
                        link_map[t, i, j] = 0.0
                elif pattern_type == 'simple':
#                     print(f"beam vector: {beam_vector}, rx vector {to_rx_j}")
#                     print(f"tx {i} to rx{j} form angle {angle}")
                    link_map[t, i, j] = antenna_gain_pattern_simple(
                        angle_deg, beamwidth_degrees, main_gain_db, side_lobe_db)
                elif pattern_type == 'realistic':
                    link_map[t, i, j] = antenna_gain_pattern_realistic(
                        angle_deg, beamwidth_degrees, main_gain_db, side_lobe_db)
    
    return link_map

# Example usage
if __name__ == "__main__":
    # Initialize positions
    tx_init = np.array([[0, 0], [100, 0], [200, 0]])
    rx_init = np.array([[400, 120], [510, 110], [420, 210]])
    
    # Generate channel data with correlation
    H, h_norm = generate_channel_data(
        num_time_steps=100,
        tx_init=tx_init,
        rx_init=rx_init,
        velocities=50,
        vibration_std=10,
        area_size=1200,
        k_factor_dB=10,
        mode='rician',
        rho_spatial=1.0,    # Spatial correlation coefficient
        rho_temporal=1.0    # Temporal correlation coefficient
    )
    
    
    # Verify temporal correlation
    temporal_corr = []
    for i in range(H.shape[1]):
        for j in range(H.shape[2]):
            h_series = H[:, i, j]
            correlation = np.corrcoef(h_series[:-1], h_series[1:])[0, 1]
            temporal_corr.append(np.abs(correlation))
    
    print(f"Average temporal correlation: {np.mean(temporal_corr):.3f}")
#     print(f"Expected temporal correlation: {rho_temporal:.3f}")
    # Calculate magnitude of complex channel values
    h_mag = np.sqrt(np.abs(H))

    # Plot channel magnitude over time for direct links (tx_i to rx_i)
    plt.figure(figsize=(12, 4))
    for i in range(h_mag.shape[1]):
        plt.plot(h_mag[:, i, i], label=f'TX{i+1}-RX{i+1}')
    plt.xlabel('Time step')
    plt.ylabel('Channel magnitude')
    plt.title('Direct channel magnitude evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()