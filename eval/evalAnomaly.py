import os
import cv2
import glob
import torch
import random
import time
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import psutil
import torch.cuda as cuda

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

def get_inference_time(model, input_tensor, num_iterations=100):
    """Calculate average inference time"""
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    
    # GPU warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Actual timing
    with torch.no_grad():
        for _ in range(num_iterations):
            starter.record()
            _ = model(input_tensor)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
    
    avg_time = sum(timings) / len(timings)
    return avg_time

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
    else:
        gpu_memory = 0
        gpu_memory_reserved = 0
    
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**2  # Convert to MB
    
    return cpu_memory, gpu_memory, gpu_memory_reserved

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
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

    # Calculate and print model size
    model_size = get_model_size(model)
    print(f"\nModel Size: {model_size:.2f} MB")
    file.write(f"\nModel Size: {model_size:.2f} MB")

    anomaly_score_list = []
    ood_gts_list = []
    
    # Get initial memory usage
    cpu_mem_start, gpu_mem_start, gpu_reserved_start = get_memory_usage()
    print(f"\nInitial Memory Usage:")
    print(f"CPU Memory: {cpu_mem_start:.2f} MB")
    print(f"GPU Memory Allocated: {gpu_mem_start:.2f} MB")
    print(f"GPU Memory Reserved: {gpu_reserved_start:.2f} MB")
    
    # Process images and measure inference time
    total_inference_time = 0
    num_images = 0
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        
        if not args.cpu:
            images = images.cuda()
        
        # Measure inference time for this image
        inference_time = get_inference_time(model, images)
        total_inference_time += inference_time
        num_images += 1
        
        with torch.no_grad():
            result = model(images)
            
        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
        
        pathGT = path.replace("images", "labels_masks")
        pathGT = osp.splitext(pathGT)[0] + ".png"
        
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
            
        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    # Get final memory usage
    cpu_mem_end, gpu_mem_end, gpu_reserved_end = get_memory_usage()
    
    # Calculate average inference time
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0
    
    # Print performance metrics
    print(f"\nPerformance Metrics:")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms per image")
    print(f"Memory Usage Delta:")
    print(f"CPU Memory: {cpu_mem_end - cpu_mem_start:.2f} MB")
    print(f"GPU Memory Allocated: {gpu_mem_end - gpu_mem_start:.2f} MB")
    print(f"GPU Memory Reserved: {gpu_reserved_end - gpu_reserved_start:.2f} MB")
    
    file.write(f"\nPerformance Metrics:")
    file.write(f"\nAverage Inference Time: {avg_inference_time:.2f} ms per image")
    file.write(f"\nMemory Usage Delta:")
    file.write(f"\nCPU Memory: {cpu_mem_end - cpu_mem_start:.2f} MB")
    file.write(f"\nGPU Memory Allocated: {gpu_mem_end - gpu_mem_start:.2f} MB")
    file.write(f"\nGPU Memory Reserved: {gpu_reserved_end - gpu_reserved_start:.2f} MB")

    # Original metrics calculation
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('\nAUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0)))
    file.close()

if __name__ == '__main__':
    main()
