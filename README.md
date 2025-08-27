# Symbol_Level_Precoding
Use QAM constructive interference to mitigate co-channel interference

To train the file, change parameters in train function, and compare the results in test file

# Files
- `generate_csi.py`: Generates synthetic CSI data with Rician fading.
- `st_resnet.py`: Defines the ST-ResNet architecture.
- `train.py`: Training loop using data augmentation over channel correlation parameters.
- 
- 'generate_corr_csi.py': Generate channel data with spatial and temporal correlation
- 'Tx_encoder.py': It contains the SLP-FANETs model and training loop which generate Symbol_level precoding scheme based on correlated csi
- 'test_model.py': Generate SNR vs BER and receiver constellation graph 

# Usage
Install all requirements packages in requirements.txt

```bash
python Tx_encoder.py
```



