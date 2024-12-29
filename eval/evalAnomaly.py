import os
import cv2
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import roc_auc_score, average_precision_score

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def maxLogit(outputs):
    """
    Calcola il logit massimo per ogni pixel nell'output del modello (prima di softmax).

    Args:
        outputs (torch.Tensor): Output del modello (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Tensor contenente il logit massimo per pixel (batch_size, height, width).
    """
    max_logits, _ = torch.max(outputs, dim=1)  # Trova il massimo logit per ogni pixel
    return max_logits


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space-separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    # Definizione soglia per rilevamento anomalie
    threshold = 1.5  # Soglia per MaxLogit (adattabile ai dati)

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    # Funzione personalizzata per caricare i pesi
    def load_my_state_dict(model, state_dict):
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

    for path in files:
        print(f"Processing image: {path}")

        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0, 3, 1, 2)

        with torch.no_grad():
            result = model(images)

        # Calcolo del MaxLogit
        max_logits = maxLogit(result).squeeze(0).cpu().numpy()

        # Mappa di punteggio delle anomalie
        anomaly_result = max_logits

        print(f"Anomaly score calculated using MaxLogit for image: {path}")

        # Applica la soglia per determinare l'anomalia
        binary_anomaly_map = (anomaly_result > threshold).astype(np.uint8)

        # Usa la mappa binaria di anomalie per decidere OOD o IN
        predicted_ood = binary_anomaly_map  # Ora usiamo questa per decidere se è OOD

        # Uso di pathlib per manipolare i percorsi delle maschere semantiche e di colore
        path_semantic = path.parent.parent / "labels_masks" / path.stem
        path_semantic = path_semantic.with_name(path_semantic.stem + "_labels_semantic.png")
        path_color = path.parent.parent / "labels_masks" / path.stem
        path_color = path_color.with_name(path_color.stem + "_labels_semantic_color.png")

        try:
            # Carica le maschere
            semantic_mask = np.array(Image.open(path_semantic))
            color_mask = np.array(Image.open(path_color))
            color_mask_gray = cv2.cvtColor(color_mask, cv2.COLOR_RGB2GRAY)

            # Combina maschera semantica e di colore
            combined_mask = np.where((semantic_mask == 2) | (color_mask_gray > 0), 1, 0)

            if 1 not in np.unique(combined_mask):
                print(f"No OOD pixels found for {path_semantic}, skipping image.")
                continue
            else:
                # Usa la mappa binaria per identificare OOD e IN-distribution
                ood_gts_list.append(combined_mask)
                anomaly_score_list.append(predicted_ood)  # Usa la mappa binaria

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    del result, anomaly_result, semantic_mask, color_mask
    torch.cuda.empty_cache()

    file.write("\n")

    if len(ood_gts_list) == 0 or len(anomaly_score_list) == 0:
        print("No valid data for evaluation. Please check your input images and labels.")
        return

    ood_gts = np.array(ood_gts_list)
    predicted_ood_maps = np.array(anomaly_score_list)
    print((ood_gts == 1))
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    # Usa la mappa binaria per determinare se un pixel è OOD o IN-distribution
    ood_out = predicted_ood_maps[ood_mask]
    ind_out = predicted_ood_maps[ind_mask]

    if len(ood_out) == 0 or len(ind_out) == 0:
        print("Error: No OOD or IND samples found.")
        return

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    # Calcolo delle metriche AUPRC e FPR@95
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc * 100.0}')
    print(f'FPR@TPR95: {fpr * 100.0}')

    file.write(('AUPRC score:' + str(prc_auc * 100.0) + '   FPR@TPR95:' + str(fpr * 100.0)))
    file.close()


if __name__ == '__main__':
    main()
