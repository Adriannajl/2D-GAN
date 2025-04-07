# Dual Decoding Generative Adversarial Networks for Infrared Image Enhancement
Official PyTorch implementation of "Dual Decoding Generative Adversarial Networks for Infrared Image Enhancement"
## Prerequisites
- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, h5py, sklearn, matplotlib, thop, opencv-python, scikit-image, torchvision
  
You can install the necessary dependencies with the following command:
```bash
pip install torch==1.1.0
pip install PyYAML tqdm tensorboardX h5py sklearn matplotlib thop opencv-python scikit-image torchvision
```
## Data Preparation
### There are 2 datasets to download:
- ImageNet Database  
  Request dataset here: [https://image-net.org/download](https://image-net.org/download)
- Sober-Drunk Database  
  Request dataset here: [https://github.com/YuYang88888/Sober-Drunk-Database](https://github.com/YuYang88888/Sober-Drunk-Database)

## Training the Model
To start training the model, simply run the following Python script. The training will use the dataset specified in the `data_path` field of the `config` dictionary.
```bash
python model.py
```
## Model Evaluation
After training, the model will be evaluated on the test dataset. It will output performance metrics such as PSNR, SSIM, and other visual assessments.


## Acknowledgements
- The implementation of [WGAN-GP](https://arxiv.org/abs/1704.00028) for stable adversarial training.

Thanks to the original authors for their work!
