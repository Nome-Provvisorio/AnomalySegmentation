import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_curve
from torch.nn.functional import softmax
from torchvision.transforms import Compose, Resize, ToTensor
import cv2

NUM_CLASSES = 20

# Define threshold for anomaly detection
threshold = 0.5  # Adjustable

def maxEntropy(outputs):
    """
    Calcola l'entropia massima per ogni pixel nell'output del modello.

    Args:
        outputs (torch.Tensor): Output del modello (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Tensor contenente l'entropia per pixel (batch_size, 1, height, width).
    """
    probabilities = softmax(outputs, dim=1)  # Probabilità per ogni classe
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1, keepdim=True)
    return entropy

def main(args):

    # Inizializza file dei risultati
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

    base_path = Path("C:/Users/vcata/Downloads/dataset_ObstacleTrack/images")
    files = list(base_path.glob("*.webp"))

    ood_gts_list = []
    anomaly_score_list = []

    for path in files:
        path = Path(path)  # Convert path to a Path object
        print(f"Processing image: {path}")

        # Caricamento immagine
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0, 3, 1, 2)

        with torch.no_grad():
            result = model(images)

        # Calcolo Max Entropy
        max_entropy = maxEntropy(result).squeeze(0).cpu().numpy()

        # Creazione della mappa binaria di anomalie
        binary_anomaly_map = (max_entropy > threshold).astype(np.uint8)

        # Caricamento della ground truth OOD
        path_semantic = path.parent.parent / "labels_masks" / path.stem
        path_semantic = path_semantic.with_name(path_semantic.stem + "_labels_semantic.png")

        try:
            semantic_mask = np.array(Image.open(path_semantic))

            # Ground truth binaria: 1 per OOD, 0 per IND
            combined_mask = (semantic_mask == 2).astype(np.uint8)

            # Aggiungi i dati a liste
            ood_gts_list.append(combined_mask)
            anomaly_score_list.append(max_entropy)

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # Calcolo delle metriche
    if len(ood_gts_list) > 0 and len(anomaly_score_list) > 0:
        ood_gts = np.concatenate([m.flatten() for m in ood_gts_list])
        anomaly_scores = np.concatenate([a.flatten() for a in anomaly_score_list])

        # Calcolo FPR@95 e AUPRC
        fpr, tpr, thresholds = roc_curve(ood_gts, anomaly_scores)
        auprc = average_precision_score(ood_gts, anomaly_scores)

        # Calcolo FPR al TPR 95%
        idx = np.where(tpr >= 0.95)[0][0]
        fpr95 = fpr[idx]

        print(f"AUPRC: {auprc * 100:.2f}%")
        print(f"FPR@95: {fpr95 * 100:.2f}%")

        file.write(('AUPRC score:' + str(auprc * 100.0) + '   FPR@TPR95:' + str(fpr95 * 100.0)))
    else:
        print("Errore: nessun dato valido per il calcolo delle metriche.")

    file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadDir', type=str, required=True, help='Directory of the model')
    parser.add_argument('--loadModel', type=str, required=True, help='Model file name')
    parser.add_argument('--loadWeights', type=str, required=True, help='Weights file name')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()
    main(args)
