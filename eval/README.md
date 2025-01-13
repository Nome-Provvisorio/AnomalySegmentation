# Functions for evaluating/visualizing the network's output

Currently there are 4 usable functions to evaluate stuff:
- eval_cityscapes_color
- eval_cityscapes_server
- eval_iou
- eval_forwardTime

## evalAnomaly.py

This code can be used to produce anomaly segmentation results on various anomaly metrics.

**Examples:**
```
python evalAnomaly.py --input '/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png'
```

## eval_cityscapes_color.py 

This code can be used to produce segmentation of the Cityscapes images in color for visualization purposes. By default it saves images in eval/save_color/ folder. You can also visualize results in visdom with --visualize flag.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_color.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_cityscapes_server.py 

This code can be used to produce segmentation of the Cityscapes images and convert the output indices to the original 'labelIds' so it can be evaluated using the scripts from Cityscapes dataset (evalPixelLevelSemanticLabeling.py) or uploaded to Cityscapes test server. By default it saves images in eval/save_results/ folder.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_server.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_iou.py 

This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val' or 'train'). For other options check the bottom side of the file.

**Examples:**
```
python eval_iou.py --datadir /home/datasets/cityscapes/ --subset val
```

## eval_forwardTime.py
This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution. 

**Options:** Option '--width' specifies the width (default: 1024). Option '--height' specifies the height (default: 512). For other options check the bottom side of the file.

**Examples:**
```
python eval_forwardTime.py
```

**NOTE**: Paper values were obtained with a single Titan X (Maxwell) and a Jetson TX1 using the original Torch code. The pytorch code is a bit faster, but cudahalf (FP16) seems to give problems at the moment for some pytorch versions so this code only runs at FP32 (a bit slower).

## evalAnomaly_E+D.py
This code can be used to produce anomaly segmentation results on various anomaly metrics and on various ways in which models are trained.

**Options:** This file have the same options of evalAnomaly.py plus the following: Option '--method' msp, maxlogit, maxentropy. Option '--E_ED' by default model towed first with encoder, then encoder and decoder (default: True). Option '--network' erfnet, enet, bisenet. Option '--one_class_selection' use only the selected class for validation (default: -1). 

**Examples:**
```
python evalAnomaly_E+D.py --input '../Validation_Dataset/RoadAnomaly21/images/*.png' --loadModel 'bisenetv2.py' --loadWeights '../trained_models/bisenet_cityscapes.pth' --method 'msp' --network bisenet
```
## eval_iou_erfnet_E+D.py 
This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

**Options:** This file have the same options of eval_iou.py plus the following: Option '--E_ED' by default model towed first with encoder, then encoder and decoder (default: True). Option '--ignore_void_label' (default: True).

**Examples:**
```
python eval_iou_erfnet_E+D.py --datadir ../datasets/ --subset val --loadWeights erfnet_cityscapes_EIML_E_ED.pth
```

## eval_iou_enet.py 
This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

**Options:** This file have the same options of eval_iou.py

**Examples:**
```
python eval_iou_enet.py --datadir ../datasets/ --subset val
```

## eval_iou_bisenet.py 
This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

**Options:** This file have the same options of eval_iou.py

**Examples:**
```
python eval_iou_bisenet.py --datadir ../datasets/ --subset val
```

