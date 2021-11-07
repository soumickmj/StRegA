# Segmentation Regularized Anomaly (StRegA)

This repository contains the code for the anomaly detection pipeline proposed. 

### Code structure

- `engine.py` and `train.py` are used to train new models with a custom data loader expected to iterate over slices of FSL segmented data on the 2D model
- `ccevae.py` contains the model code and uses parts from `ae_bases.py`, `ce_noise.py` and `helpers.py`
- `Pipeline.ipynb` shows the entire StRegA pipeline including post-processing.
- The `dataloaders` folder has some examples of the dataloader that was used during training and validation

### Checkpoint

The model checkpoint `brain.ptrh` can be loaded with the same model with `torch.load` function. This was trained on IXI + MOOD T1, T2 and Proton Density Images that were segmented with FSL. 
