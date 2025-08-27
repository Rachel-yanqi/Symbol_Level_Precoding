## Precoding based on constructive interference area
Use PSK/QAM constructive interference to mitigate multi-user interference 

To train the file, change parameters in train function, and compare the results in test file

## Files
- `generate_corr_csi.py`: Generate channel data with spatial and temporal correlation
- `Tx_encoder.py`: It contains the SLP-FANETs model and training loop which generate Symbol_level precoding scheme based on correlated csi
- `test_model.py`: Generate SNR vs BER and receiver constellation graph
- `requirements.txt` List all installed packages for this project

## Usage
`Install all requirements packages in requirements.txt`

```bash
python Tx_encoder.py
```





