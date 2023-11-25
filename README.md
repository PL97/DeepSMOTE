# DeepSMOTE
This repository provides a *unoffical* pytorch implementation of "DeepSMOTE: Fusing Deep Learning and SMOTE
for Imbalanced Data"

## Prerequsits
```bash
conda env create -f environment.yml
conda activate DeepSMOTE
```

## demo of results 
### naive autoencoder
***left:*** origional image ***right:*** reconstruction          |  sythetic images using smote
:-------------------------:|:-------------------------:
<img src="figures/version_0/recon_img_gallery.png" width="120" />  |  <img src="figures/version_0/synthetic.png" width="300" /> 

### autoencoder with random permutation
***left:*** origional image ***right:*** reconstruction          |  sythetic images using smote
:-------------------------:|:-------------------------:
<img src="figures/version_1/recon_img_gallery.png" width="120" />  |  <img src="figures/version_1/synthetic.png" width="300" /> 

## Run the code
```bash
## train model
python main.py --train

## test model (show reconstructions and sythetic images)
python main.py
```

## To Do
### dataset
- [x] [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ ] [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
### network archicture
- [x] hand-craft CNN
- [ ] UNet

:star2: Please star this repo if you find it helpful!

If you find this git repo useful, please consider citing the associated paper and ***acknowledge our implementation*** using the snippet below:
```bibtex
@article{peng2022imbalanced,
  title={Imbalanced Classification in Medical Imaging},
  author={Peng, Le and Travadi, Yash and Zhang, Rui and Cui, Ying and Sun, Ju},
  journal={arXiv preprint arXiv:2210.12234},
  year={2022}
}
```
