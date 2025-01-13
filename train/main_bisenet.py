import numpy as np
import torch
import os
import random
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageOps
from transform import Relabel, ToLabel
from dataset import cityscapes
from bisenetv2 import BiSeNetV2

NUM_CLASSES = 20

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass

    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)
        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)
            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   
        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

def calculate_iou(predictions, targets, num_classes):
    ious = torch.zeros(num_classes)
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union != 0:
            ious[cls] = intersection / union
    return ious

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50):
    for epoch in range(num_epochs):
        print("----- TRAINING - EPOCH", epoch, "-----")
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader):
            images = images.cuda()
            targets = targets.squeeze(1).cuda()  # Riduce la dimensione del target
            # Forward pass
            outputs = model(images)[0]
            loss = criterion(outputs, targets)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.cuda()
                targets = targets.squeeze(1).cuda()
                outputs = model(images)[0]
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                total_iou += calculate_iou(preds, targets, NUM_CLASSES).mean().item()
        print(f"Validation Mean IoU: {total_iou / len(val_loader)}")

def main():
    datadir = "../datasets"
    num_workers = 4
    batch_size = 4
    height = 512
    num_epochs = 100
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.8149201869965	
    weight[1] = 6.9850029945374	
    weight[2] = 3.7890393733978	
    weight[3] = 9.9428062438965	
    weight[4] = 9.7702074050903	
    weight[5] = 9.5110931396484	
    weight[6] = 10.311357498169	
    weight[7] = 10.026463508606	
    weight[8] = 4.6323022842407	
    weight[9] = 9.5608062744141	
    weight[10] = 7.8698215484619	
    weight[11] = 9.5168733596802	
    weight[12] = 10.373730659485	
    weight[13] = 6.6616044044495	
    weight[14] = 10.260489463806	
    weight[15] = 10.287888526917	
    weight[16] = 10.289801597595	
    weight[17] = 10.405355453491	
    weight[18] = 10.138095855713
    weight[19] = 0
    weight = weight.cuda()
    # Trasformazione per input + Dataset + Dataloader
    co_transform = MyCoTransform(False, augment=True, height=height)
    co_transform_val = MyCoTransform(False, augment=False, height=height)
    dataset_train = cityscapes(datadir, co_transform, 'train')
    dataset_val = cityscapes(datadir, co_transform_val, 'val')
    train_loader = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    model = BiSeNetV2(NUM_CLASSES)
    model.cuda()
    # Ottimizzatore e criterio di perdita
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = CrossEntropyLoss2d(weight)
    # Esegui il training
    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs)
    torch.save(model.state_dict(), "bisenet_cityscapes.pth")

if __name__ == '__main__':
    main()


#python main_bisenet.py