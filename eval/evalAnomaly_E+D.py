# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from erfnet import ERFNet
from enet import ENet
from bisenetv2 import BiSeNetV2

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
'''
import psutil
import time
from fvcore.nn import FlopCountAnalysis

def calculate_flops(model, input_tensor):
    # Calcola i FLOPs per il modello e un singolo input
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()

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
'''
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
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default="msp") #msp, maxlogit, maxentropy
    parser.add_argument('--E_ED', default=True) #di default modello trainato prima con encoder, poi encoder e decoder
    parser.add_argument('--network', default="erfnet") #erfnet, enet, bisenet
    parser.add_argument('--one_class_selection', default=-1) #use only the selected class for validation
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []
    '''
    total_flops = 0
    # Initialize timing metrics
    total_inference_time = 0
    num_processed_images = 0
    '''
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    '''
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
    '''
    if args.network == "erfnet":
        model = ERFNet(NUM_CLASSES)
    elif args.network == "enet":
        model = ENet(NUM_CLASSES)
    elif args.network == "bisenet":
        model = BiSeNetV2(NUM_CLASSES)
    else:
        print("Error on chosing the network")
        return -1

    if args.E_ED:
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
    else:
        if (not args.cpu):
            model.cuda()
        model.load_state_dict(torch.load(weightspath, map_location=lambda storage, loc: storage))
    model.eval()
    print ("Model and weights LOADED successfully")
    '''
    # Calculate and log model size
    model_size = get_model_size(model)
    print(f"\nModel Size: {model_size:.2f} MB")
    file.write(f"\nModel Size: {model_size:.2f} MB")
    '''
    VOID_CLASS_INDEX = args.one_class_selection
    #step = 0
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)[0]
        '''
        if step == 2:
            pred = torch.argmax(result, dim=0)
            prediction_image = pred.cpu().numpy() # Porta l'immagine su CPU e converti in numpy
            import matplotlib.pyplot as plt

            # Mostra l'immagine sorgente
            plt.subplot(1, 2, 1)
            plt.imshow(Image.open(path).convert('RGB'))
            plt.title("Source Image")
            plt.axis('off')

             # Mostra la mappa di segmentazione
            plt.subplot(1, 2, 2)
            plt.imshow(prediction_image, cmap='tab20')
            plt.colorbar()
            plt.title("Predicted Segmentation")
            plt.axis('off')

            plt.show()
        step += 1
        '''
        '''
        # Calcola i FLOPs per l'immagine corrente
        image = Image.open(path).convert('RGB')
        image_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float().permute(0, 3, 1, 2).cuda()

        flops = calculate_flops(model, image_tensor)
        total_flops += flops  # Somma i FLOPs totali
        if not args.cpu:
            images = images.cuda()

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            result = model(images)[0]
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_inference_time += inference_time
        num_processed_images += 1
        '''
        if VOID_CLASS_INDEX != -1:
            # Calcolo delle probabilità softmax
            probabilities = torch.softmax(result.squeeze(0), dim=0).data.cpu().numpy()  # Shape: [num_classes, H, W]
            # Seleziona solo le probabilità della classe Void
            void_probabilities = probabilities[VOID_CLASS_INDEX]  # Shape: [H, W]
            # Calcola il punteggio di anomalia MSP
            anomaly_result = 1.0 - void_probabilities  # Anomalia come 1 - P(Void)
        else:
            if args.method == "msp":
                ## QUESTO è MSP
                probabilities = torch.softmax(result.squeeze(0), dim=0).data.cpu().numpy()
                anomaly_result = 1.0 - np.max(probabilities, axis=0)
            elif args.method == "maxentropy":
                ## QUESTO è MAXENTROPY
                probabilities = torch.softmax(result.squeeze(0), dim=0).data.cpu().numpy()
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=0)  # Aggiungi un epsilon per evitare log(0)
                anomaly_result = entropy
            elif args.method == "maxlogit":
                #QUESTO è MAXLOGIT
                # Calcola il logit massimo per ogni pixel (prima di softmax)
                logits = result.squeeze(0).data.cpu().numpy()
                anomaly_result = -np.max(logits, axis=0)
            else:
                print("Errore nella scelta del method")
                return -1

        pathGT = path.replace("images", "labels_masks")   
        pathGT = osp.splitext(pathGT)[0] + ".png"
        
        #print("prima dell'if ",pathGT)
        
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png") 
            
        #print("dopo l'if ",pathGT)
        
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
    '''
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
    if num_processed_images > 0:
        avg_flops_per_inference = total_flops / num_processed_images
        print(f"FLOPs medi per inferenza: {avg_flops_per_inference:.4f}")
    else:
        print("Nessuna immagine processata. Impossibile calcolare la media dei FLOPs.")
    '''
    # Process results
    file.write( "\n")

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

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()


#python evalAnomaly_E+D.py --input '../Validation_Dataset/RoadAnomaly21/images/*.png' --loadModel 'bisenetv2.py' --loadWeights '../trained_models/bisenet_cityscapes.pth' --method 'msp'
#python evalAnomaly_E+D.py --input '../Validation_Dataset/RoadAnomaly21/images/*.png' --loadModel 'erfnet.py' --loadWeights '../trained_models/erfnet_cityscapes_EIML+CE.pth'