# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        #nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--metric', choices=['msp', 'maxentropy', 'maxlogit', 'msp-temperature'], default='msp', help="Tipo di metrica da utilizzare.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperatura per la normalizzazione delle probabilità (default è 1.0).")
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()


    base_path = Path(args.input)
    files = list(base_path.glob("*.*"))
    print("ptha", base_path)
    for path in files:
        print(f"Processing image: {path}")  # Log percorso immagine
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)


        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)

        # Seleziona la metrica in base all'argomento
        if args.metric == 'msp-temperature':
            # MSP (Maximum Softmax Probability)
            scaled_result = result / args.temperature
            probabilities = torch.softmax(scaled_result.squeeze(0), dim=0).data.cpu().numpy()
            anomaly_result = 1.0 - np.max(probabilities, axis=0)

        elif args.metric == 'msp':
            # MSP (Maximum Softmax Probability)
            anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)

        elif args.metric == 'maxentropy':
            # Entropia massima
            probabilities = torch.softmax(result.squeeze(0), dim=0).data.cpu().numpy()
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=0)  # Evita log(0) con epsilon
            anomaly_result = entropy

        elif args.metric == 'maxlogit':
            # Maximum Logit - take negative of maximum logit value
            # Higher negative values indicate higher anomaly scores
            logits = result.squeeze(0).data.cpu().numpy()
            anomaly_result = -np.max(logits, axis=0)

        pathGT = path.parent.parent / "labels_masks" / path.stem
        pathGT = pathGT.with_name(pathGT.stem + ".png")
        if "RoadObsticle21" in str(pathGT):
            pathGT = pathGT.with_suffix(".png")
        if "fs_static" in str(pathGT):
            pathGT = pathGT.with_suffix(".png")
        if "RoadAnomaly" in str(pathGT):
            pathGT = pathGT.with_suffix(".png")

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in str(pathGT):
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in str(pathGT):
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in str(pathGT):
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if "fs" in str(pathGT):
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if "FS" in str(pathGT):
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    if len(ood_gts) == 0 or len(anomaly_scores) == 0:
        print("No valid anomaly data found. Skipping evaluation.")
        return
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    if len(ood_out) == 0 or len(ind_out) == 0:
        print("No valid OOD or IND data available. Skipping evaluation.")
        return
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    # Safeguard for empty concatenation
    if len(val_label) == 0 or len(val_out) == 0:
        print("Combined data is empty. Skipping evaluation.")
        return
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()
