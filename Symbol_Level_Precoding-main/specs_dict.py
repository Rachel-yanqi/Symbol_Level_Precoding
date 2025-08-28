# hyperparameters.py

# Define all hyperparameters in a dictionary
specs_dict = {
    "num_samples": 10000,
    "num_users": 2,
    "constellation_size": 4,  # 4, 8, 16
    "tx_init": [[0, 0], [100, 0]],  # Initial positions of transmitters
    "rx_init": [[100, 200], [210, 110]],  # Initial positions of receivers
    "training_opt":{
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "step_lr",
        "step_size": 10,
        "gamma": 0.1,       # weigh decay factor
        "batch_size": 32,
        "num_epochs": 10,
        "file_path": "./saved_model/my_file_name.pth",
    },
    "channel_data":{
        "velocities": 0,  # User velocities in m/s
        "vibration_std": 10,
        "area_size": 200,  # Size of the area that users csi has spatial correlation
        "type": "rician",  
        "k_factor_db": 10,  # Rician K-factor in dB
        "mismatch_deg": 5,  # Antenna mismatch in degrees [None, 1, 5]
    },
    
}

