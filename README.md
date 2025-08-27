# Symbol_Level_Precoding
Use QAM constructive interference to mitigate co-channel interference

To train the file, change parameters in train function, and compare the results in test file

# Files
-'generate_corr_csi.py': Generate channel data with spatial and temporal correlation
-'Tx_encoder.py': It contains the SLP-FANETs model and training loop which generate Symbol_level precoding scheme based on correlated csi
-'test_model.py': Generate SNR vs BER and receiver constellation graph 

# Usage
Install all requirements packages in requirements.txt

```bash
python train.py
```

