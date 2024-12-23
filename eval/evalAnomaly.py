# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
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
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
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
    print("Model and weights LOADED successfully")
    model.eval()

    from pathlib import Path
    base_path = Path("C:/Users/vcata/Downloads/dataset_ObstacleTrack/images")
    files = list(base_path.glob("*.webp"))
    counter = 0
    for path in files:
        # counter += 1
        # if counter > 2:
        #     break
        path = Path(path)  # Converte il percorso in un oggetto Path
        print(f"Elaborazione immagine: {path}")  # Log percorso immagine

        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0, 3, 1, 2)

        with torch.no_grad():
            result = model(images)

        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
        print("Parent: ", path.parent.parent)

        # Usa pathlib per manipolare il percorso della maschera originale
        pathGT = path.parent.parent / "labels_masks" / path.stem
        pathGT = pathGT.with_name(pathGT.stem + "_labels_semantic.png")
        print(f"Percorso iniziale della maschera: {pathGT}")  # Verifica percorso iniziale della maschera

        # Aggiungi gestione per immagine con ostacolo
        pathColor = path.parent / (path.stem + "_color" + path.suffix)
        print(f"Percorso immagine con ostacolo: {pathColor}")

        try:
            mask = Image.open(pathGT)  # Carica la maschera originale
            ood_gts = np.array(mask)
            print(f"Valori iniziali nella maschera: {np.unique(ood_gts)}")  # Valori iniziali della maschera

            if "dataset_ObstacleTrack" in str(pathGT):
                ood_gts = np.where((ood_gts == 2), 1, ood_gts)

            print(f"Valori dopo trasformazione: {np.unique(ood_gts)}")
            print(ood_gts.shape)  # Valori trasformati della maschera

            if 1 not in np.unique(ood_gts):
                print(f"Nessun pixel OOD trovato per {pathGT}, salto immagine.")  # Log in caso di maschera senza OOD
                continue
            else:
                ood_gts_list.append(ood_gts)
                print("Append")
                anomaly_score_list.append(anomaly_result)

            # Verifica immagine con ostacolo e aggiungi logica se necessario
            if pathColor.exists():
                print(f"Trovata immagine con ostacolo: {pathColor}")
                images_color = torch.from_numpy(np.array(Image.open(pathColor).convert('RGB'))).unsqueeze(0).float()
                images_color = images_color.permute(0, 3, 1, 2)

                with torch.no_grad():
                    result_color = model(images_color)

                anomaly_result_color = 1.0 - np.max(result_color.squeeze(0).data.cpu().numpy(), axis=0)
                anomaly_score_list.append(anomaly_result_color)
                print("Aggiunto risultato immagine con ostacolo.")

        except Exception as e:
            print(f"Errore nell'elaborazione di {pathGT}: {e}")  # Log in caso di errore
            continue

    del result, anomaly_result, ood_gts, mask
    torch.cuda.empty_cache()

    file.write("\n")

    if len(ood_gts_list) == 0 or len(anomaly_score_list) == 0:
        print("ood_gts_list: ", ood_gts_list)
        print("anomaly_score_list: ", anomaly_score_list)
        print("No valid data for evaluation. Please check your input images and labels.")
        return

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    if ood_gts.size == 0 or anomaly_scores.size == 0:
        print("Error: No valid data for evaluation.")
        return

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    if len(ood_out) == 0 or len(ind_out) == 0:
        print("Error: No OOD or IND samples found.")
        return

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc * 100.0}')
    print(f'FPR@TPR95: {fpr * 100.0}')

    file.write(('    AUPRC score:' + str(prc_auc * 100.0) + '   FPR@TPR95:' + str(fpr * 100.0)))
    file.close()

if __name__ == '__main__':
    main()
