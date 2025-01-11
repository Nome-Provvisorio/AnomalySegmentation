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
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import psutil
import time

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

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
    else:
        gpu_memory = 0
        gpu_memory_reserved = 0
    
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**2  # MB
    
    return cpu_memory, gpu_memory, gpu_memory_reserved

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--metric', choices=['msp', 'maxentropy', 'maxlogit', 'msp-temperature'], default='msp')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    # Record initial memory state
    cpu_mem_start, gpu_mem_start, gpu_reserved_start = get_memory_usage()
    print("\nInitial Memory State:")
    print(f"CPU Memory: {cpu_mem_start:.2f} MB")
    print(f"GPU Memory Allocated: {gpu_mem_start:.2f} MB")
    print(f"GPU Memory Reserved: {gpu_reserved_start:.2f} MB")
    file.write(f"\n\nInitial Memory State:")
    file.write(f"\nCPU Memory: {cpu_mem_start:.2f} MB")
    file.write(f"\nGPU Memory Allocated: {gpu_mem_start:.2f} MB")
    file.write(f"\nGPU Memory Reserved: {gpu_reserved_start:.2f} MB")

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
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
    
    # Calculate and log model size
    model_size = get_model_size(model)
    print(f"\nModel Size: {model_size:.2f} MB")
    file.write(f"\nModel Size: {model_size:.2f} MB")
    
    model.eval()

    base_path = Path(args.input)
    files = list(base_path.glob("*.*"))
    print("Path:", base_path)
    
    anomaly_score_list = []
    ood_gts_list = []
    
    # Initialize timing metrics
    total_inference_time = 0
    num_processed_images = 0
    
    for path in files:
        print(f"Processing image: {path}")
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        if not args.cpu:
            images = images.cuda()
            
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            result = model(images)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_inference_time += inference_time
        num_processed_images += 1

        # Calculate anomaly scores based on selected metric
        if args.metric == 'msp-temperature':
            probabilities = torch.softmax(result.squeeze(0) / args.temperature, dim=0).data.cpu().numpy()
            anomaly_result = 1.0 - np.max(probabilities, axis=0)
        elif args.metric == 'msp':
            probabilities = torch.softmax(result.squeeze(0), dim=0).data.cpu().numpy()
            anomaly_result = 1.0 - np.max(probabilities, axis=0)
        elif args.metric == 'maxentropy':
            probabilities = torch.softmax(result.squeeze(0), dim=0).data.cpu().numpy()
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=0)
            anomaly_result = entropy
        elif args.metric == 'maxlogit':
            logits = result.squeeze(0).data.cpu().numpy()
            anomaly_result = -np.max(logits, axis=0)

        pathGT = path.parent.parent / "labels_masks" / path.stem
        pathGT = pathGT.with_name(pathGT.stem + ".png")
        
        for dataset in ["RoadObsticle21", "fs_static", "RoadAnomaly", "fs", "FS"]:
            if dataset in str(pathGT):
                pathGT = pathGT.with_suffix(".png")

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        # Process different dataset formats
        if "RoadAnomaly" in str(pathGT):
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        elif "LostAndFound" in str(pathGT):
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        elif "Streethazard" in str(pathGT):
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)
        elif "fs" in str(pathGT) or "FS" in str(pathGT):
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
            
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    # Get final memory state
    cpu_mem_end, gpu_mem_end, gpu_reserved_end = get_memory_usage()
    
    # Calculate and log performance metrics
    avg_inference_time = total_inference_time / num_processed_images if num_processed_images > 0 else 0
    print("\nPerformance Metrics:")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms/image")
    print(f"Total Images Processed: {num_processed_images}")
    print("\nMemory Usage Delta:")
    print(f"CPU Memory: {cpu_mem_end - cpu_mem_start:.2f} MB")
    print(f"GPU Memory Allocated: {gpu_mem_end - gpu_mem_start:.2f} MB")
    print(f"GPU Memory Reserved: {gpu_reserved_end - gpu_reserved_start:.2f} MB")
    
    file.write("\n\nPerformance Metrics:")
    file.write(f"\nAverage Inference Time: {avg_inference_time:.2f} ms/image")
    file.write(f"\nTotal Images Processed: {num_processed_images}")
    file.write("\nMemory Usage Delta:")
    file.write(f"\nCPU Memory: {cpu_mem_end - cpu_mem_start:.2f} MB")
    file.write(f"\nGPU Memory Allocated: {gpu_mem_end - gpu_mem_start:.2f} MB")
    file.write(f"\nGPU Memory Reserved: {gpu_reserved_end - gpu_reserved_start:.2f} MB")

    # Process results if we have valid data
    if len(ood_gts_list) > 0 and len(anomaly_score_list) > 0:
        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)
        
        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        if len(ood_out) > 0 and len(ind_out) > 0:
            ood_label = np.ones(len(ood_out))
            ind_label = np.zeros(len(ind_out))
            
            val_out = np.concatenate((ind_out, ood_out))
            val_label = np.concatenate((ind_label, ood_label))

            if len(val_label) > 0 and len(val_out) > 0:
                prc_auc = average_precision_score(val_label, val_out)
                fpr = fpr_at_95_tpr(val_out, val_label)

                print(f'\nAUPRC score: {prc_auc*100.0}')
                print(f'FPR@TPR95: {fpr*100.0}')

                file.write(f'\nAUPRC score: {prc_auc*100.0}')
                file.write(f'\nFPR@TPR95: {fpr*100.0}')
    else:
        print("No valid data for evaluation")
        file.write("\nNo valid data for evaluation")

    file.close()

if __name__ == '__main__':
    main()
