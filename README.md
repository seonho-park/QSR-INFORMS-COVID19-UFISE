### TODO
- [x] test routine: accuracy, F1 score, AUC score
- [x] gathering other dataset for lung segmentation
- [x] lung segment augmentation 
- [x] split test/training set based on patient ID
- [ ] weighted binary cross entropy
- [ ] contrastive self-supervised learning (CSSL) loss 
- [ ] implement other CNN architectures

### Current Performance
Tested on AUG 8 2020
- AUROC: 0.9939
- AUPR: 0.9949
- Accuracy: 0.9450
- F1 score: 0.9455

Tested on Aug 9 2020
- AUROC: 0.7430 | AUPR: 0.6281 | F1-score: 0.6549 | Accuracy: 0.6638
- This is lower than previous result, mainly due to using patient IDs when splitting dataset into training/test datasets

## Requirements
- check requirements.txt

## Lung Segmentation
- Download lung segmentation CT data from https://www.kaggle.com/kmader/finding-lungs-in-ct-data/
- Used ResNet-18 based UNet. refer to https://github.com/usuyama/pytorch-unet
- train: using lungseg.py
- for deploying lung segmentation outputs of the data: deploy_lungseg.py

## Training

## Testing
