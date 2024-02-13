#  <p align="center"> _Anatomically-aware Uncertainty for Semi-supervised Image Segmentation_
## <p align="center"> _MedIA 2024_ [[paper](https://arxiv.org/pdf/2310.16099.pdf)], _MICCAI 2022_  [[paper](https://arxiv.org/pdf/2203.05682.pdf)] [[presentation](https://github.com/adigasu/Labeling_Representations/blob/main/Files/Labeling%20representation.pdf)] [[poster](https://github.com/adigasu/Labeling_Representations/blob/main/Files/MICCAI2022_poster.pdf)]

**TL;DR:** A novel way to estimate the uncertainty maps using anatomically-aware representation prior in order to guide the segmentation model in a low-data regime.

<p align="center">  <img src = 'Files/Anatomical_rep_Arch.png' height = '300px'>

**Keywords:** Semi-Supervised learning; Anatomically-aware Representation; Labeling Representation; Image Segmentation; Uncertainty.


### Dependencies
This code depends on the following libraries:

- Pytorch (1.8.0+cu111)
- Python >= 3.8
- tensorboardX
- some basic libraries: numpy, glob, skimage, matplotlib, tqdm...

### Datasets
- [LA, 2018](https://github.com/yulequan/UA-MT/tree/master/data)
- [FLARE, 2021](https://flare.grand-challenge.org/FLARE21/)

### Training
Training of our approach involves two steps:

1) DAE (Denoising Autoencoder) model training with available labels
```
cd code_DAE
python train_DAE.py --exp DAE_L10 --nb_labels 26 --total_labels 260 --emb_dim 512
```

2) Segmentation model training with DAE under limited labels 
```
cd code_DAE
python train_Abdomen_meanteacher_DAE_certainty.py --exp L10_r1 --nb_labels 26 --total_labels 260 --model_DAE 'DAE_L10/model.pth' --emb_dim 512
```

### Testing
```
Coming soon...
```

### Citation
Please cite our paper if you find this code or our work useful for your research.
```
@article{adiga2023anatomically,
  title={Anatomically-aware Uncertainty for Semi-supervised Image Segmentation},
  author={Adiga V, Sukesh and Dolz, Jose and Lombaert, Herve},
  journal={Medical Image Analysis (MedIA)},
  year={2024}
}
```
```
@article{adiga2022leveraging,
  title={Leveraging Labeling Representations in Uncertainty-based Semi-supervised Segmentation},
  author={Adiga V, Sukesh and Dolz, Jose and Lombaert, Herve},
  journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2022}
}
```

### References
- Uncertainty-aware Self-ensembling Model for Semi-supervised (UAMT) [[paper](https://arxiv.org/abs/1907.07034)][[code](https://github.com/yulequan/UA-MT)]
- Semi-supervised Learning for Medical Image Segmentation (SSL4MIS) [[paper](https://arxiv.org/abs/2012.07042)][[code](https://github.com/HiLab-git/SSL4MIS/tree/master/code)]

#### Any questions?
```
For more information, please get in touch with Sukesh Adiga (sukesh.adiga@gmail.com).
```

#### License
This project is licensed under the terms of the MIT license. 
