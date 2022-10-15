# **S**egmen**t**ation **Reg**ularised **A**nomaly (StRegA)

Official code of the paper "StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding Variational Autoencoder" (https://doi.org/10.1016/j.compbiomed.2022.106093 and https://arxiv.org/abs/2201.13271).

This was first presented at ISMRM-ESMRMB 2022, London.
Abstract available on RG: https://www.researchgate.net/publication/358357400_StRegA_Unsupervised_Anomaly_Detection_in_Brain_MRIs_using_Compact_Context-encoding_Variational_Autoencoder

The name "StRegA" is inspired by the name of the Italian herb liquore with saffron - Strega (following the tradition of namming MR-related products with name of alchoholic drinks or liquores.

## Information regarding this repo

### Code structure

- `engine.py` and `train.py` are used to train new models with a custom data loader expected to iterate over slices of FSL segmented data on the 2D model
- `ccevae.py` contains the model code and uses parts from `ae_bases.py`, `ce_noise.py` and `helpers.py`
- `Pipeline.ipynb` shows the entire StRegA pipeline including post-processing.
- The `dataloaders` folder has some examples of the dataloader that was used during training and validation

### Checkpoint

The model checkpoint `brain.ptrh` can be loaded with the same model with `torch.load` function. This was trained on IXI + MOOD T1, T2 and Proton Density Images that were segmented with FSL. 

## Contacts

Please feel free to contact me for any questions or feedback:

[soumick.chatterjee@ovgu.de](mailto:soumick.chatterjee@ovgu.de)

[contact@soumick.com](mailto:contact@soumick.com)

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite either or both of the following in your publications:

> [Soumick Chatterjee, Alessandro Sciarra, Max Dünnwald, Pavan Tummala, Shubham Kumar Agrawal, Aishwarya Jauhari, Aman Kalra, Steffen Oeltze-Jafra, Oliver Speck, Andreas Nürnberger: StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding Variational Autoencoder (Computers in Biology and Medicine, Oct 2022)](https://doi.org/10.1016/j.compbiomed.2022.106093)

BibTeX entry:

```bibtex
@article{chatterjee2022strega,
  title={StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding Variational Autoencoder},
  author={Chatterjee, Soumick and Sciarra, Alessandro and D{\"u}nnwald, Max and Tummala, Pavan and Agrawal, Shubham Kumar and Jauhari, Aishwarya and Kalra, Aman and Oeltze-Jafra, Steffen and Speck, Oliver and N{\"u}rnberger, Andreas},
  journal={Computers in Biology and Medicine},
  pages={106093},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2022.106093}
}
}
```

The complete manuscript is also on ArXiv:-
> [Soumick Chatterjee, Alessandro Sciarra, Max Dünnwald, Pavan Tummala, Shubham Kumar Agrawal, Aishwarya Jauhari, Aman Kalra, Steffen Oeltze-Jafra, Oliver Speck, Andreas Nürnberger: StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding Variational Autoencoder (arXiv:2201.13271
, Jan 2022)](https://arxiv.org/abs/2201.13271)

The ISMRM-ESMRMB 2022 abstract:-

> [Soumick Chatterjee, Alessandro Sciarra, Max Dünnwald, Pavan Tummala, Shubham Kumar Agrawal, Aishwarya Jauhari, Aman Kalra, Steffen Oeltze-Jafra, Oliver Speck, Andreas Nürnberger: StRegA: Unsupervised Anomaly Detection in Brain MRIs using Compact Context-encoding Variational Autoencoder (ISMRM-ESMRMB 2022, May 2022)](https://www.researchgate.net/publication/358357668_Multi-scale_UNet_with_Self-Constructing_Graph_Latent_for_Deformable_Image_Registration)

BibTeX entry:


```bibtex
@inproceedings{mickISMRM22strega,
      author = {Chatterjee, Soumick and Sciarra, Alessandro and D{\"u}nnwald, Max and Tummala, Pavan and Agrawal, Shubham Kumar and Jauhari, Aishwarya and Kalra, Aman and Oeltze-Jafra, Steffen and Speck, Oliver and N{\"u}rnberger, Andreas},
      year = {2022},
      month = {05},
      pages = {0172},
      title = {StRegA: Unsupervised Anomaly Detection in Brain MRIs using Compact Context-encoding Variational Autoencoder},
      booktitle={ISMRM-ESMRMB 2022}
}
```
Thank you so much for your support.
