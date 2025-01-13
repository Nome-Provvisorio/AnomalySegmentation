# Real-Time-Anomaly-Segmentation [[Course Project](https://docs.google.com/document/d/1ElljsAprT2qX8RpePSQ3E00y_3oXrtN_CKYC6wqxyFQ/edit?usp=sharing)]
This repository provides a starter-code setup for the Real-Time Anomaly Segmentation project of the Machine Learning Course. It consists of the code base for training ERFNet, ENet and BiseNetV2 on the Cityscapes dataset and perform anomaly segmentation.

## Packages
For instructions, please refer to the README in each folder:

* [train](train) contains tools for training the network for semantic segmentation.
* [eval](eval) contains tools for evaluating/visualizing the network's output and performing anomaly segmentation.
* [imagenet](imagenet) Contains script and model for pretraining ERFNet's encoder in Imagenet.
* [trained_models](trained_models) Contains the trained models used in the papers.
* [img](img) Contains the image of the models trained. 

## Requirements:

* [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0). 
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)
* **For testing the anomaly segmentation model**: Road Anomaly, Road Obstacle, and Fishyscapes dataset. All testing images are provided here [Link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Anomaly Inference:
* The repo provides some pre-trained models that can be used to perform anomaly segmentation on test anomaly datasets:
  * bisenet_cityscapes.pth: bisenet trained whit the CE loss on 50 epoch
  * enet_cityscapes.pth: enet trained whit the CE loss on 50 epoch
  * erfnet_cityscapes_EIML_E_ED.pth: erfnet trained whit the EIML loss training first the encoder and after the encoder and decoder on 50 epoch
  * erfnet_cityscapes_EIML.pth: erfnet trained whit the EIML loss on 50 epoch
  * erfnet_cityscapes_EIML+CE_E_ED.pth: erfnet trained whit the EIML+CE loss training first the encoder and after the encoder and decoder on 50 epoch
  * erfnet_cityscapes_EIML+CE.pth: erfnet trained whit the EIML+CE loss on 50 epoch
  * erfnet_cityscapes_EIML+FL_E_ED.pth: erfnet trained whit the EIML+FL loss training first the encoder and after the encoder and decoder on 50 epoch
  * erfnet_cityscapes_EIML+FL.pth: erfnet trained whit the EIML+FL loss on 50 epoch
  * erfnet_encoder_pretrained.pth.tar: erfnet only encoder trained whit the CE loss on 150 epoch
  * erfnet_pretrained.pth: erfnet trained whit the CE loss training first the encoder and after the encoder and decoder on 150 epoch
  * model_best_combined_CROSS_LOGIT.pth: erfnet trained whit the CE+LN loss training first the encoder and after the encoder and decoder on 50 epoch
  * model_best_combined_FOCAL_LOGIT.pth: erfnet trained whit the FL+LN loss training first the encoder and after the encoder and decoder on 50 epoch
  * model_best_enc_enhenced.pth.tar: erfnet only encoder trained whit the CE loss on 50 epoch
  * model_best_log_norm.pth: erfnet trained whit the LN loss training first the encoder and after the encoder and decoder on 50 epoch
  * model_encoder_best_enhanced.pth: erfnet only encoder trained whit the CE loss on 50 epoch

* Anomaly Inference Command:```python evalAnomaly.py --input '/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png```. Change the dataset path ```'/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png```accordingly.
