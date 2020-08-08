### TODO
- [x] test routine: accuracy, F1 score, AUC score
- [x] gathering other dataset for lung segmentation
- [x] lung segment augmentation 
- [ ] weighted binary cross entropy
- [ ] split test/training set based on patient ID
- [ ] contrastive self-supervised learning (CSSL) loss 

### Current Performance
approximately....
- AUROC: 0.9841
- AUPR: 9755
- Accuracy: 0.95
- F1 score: 0.95

## Requirements
- check requirements.txt

## Lung Segmentation
- Download lung segmentation CT data from https://www.kaggle.com/kmader/finding-lungs-in-ct-data/
- Used ResNet-18 based UNet. refer to https://github.com/usuyama/pytorch-unet
- train: using lungseg.py
- for deploying lung segmentation outputs of the data: deploy_lungseg.py

## Training

## Testing
