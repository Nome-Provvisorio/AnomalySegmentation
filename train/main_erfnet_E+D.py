import numpy as np
import torch
import os
import random
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageOps
from transform import Relabel, ToLabel
from matplotlib import pyplot as plt
from torch.optim import Adam
from dataset import cityscapes
import importlib
#from erfnet import ERFNet

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

class EnhancedIsotropyMaximizationLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon (float): Piccolo valore per la stabilità numerica durante il calcolo del log.
        """
        super(EnhancedIsotropyMaximizationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, embeddings):
        """
        Args:
            embeddings (torch.Tensor): Tensore di dimensione [batch_size, embedding_dim],
                                       dove embedding_dim è la dimensione delle rappresentazioni.
        Returns:
            torch.Tensor: Loss scalare per massimizzare l'isotropia.
        """
        # Normalizzazione delle embedding per ottenere vettori unitari
        embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + self.epsilon)
        # Calcolo della matrice di covarianza
        batch_mean = embeddings.mean(dim=0, keepdim=True)  # [1, embedding_dim]
        centered_embeddings = embeddings - batch_mean
        covariance_matrix = centered_embeddings.t() @ centered_embeddings  # [embedding_dim, embedding_dim]
        # Calcolo della varianza diagonale e della distanza media tra le embedding
        diag_variance = torch.diag(covariance_matrix)  # Varianza delle singole coordinate
        mean_distance = torch.norm(centered_embeddings.unsqueeze(1) - centered_embeddings.unsqueeze(0), dim=2).mean()
        # Loss: combinazione di varianza inversa e distanza media
        loss = diag_variance.mean().reciprocal() + mean_distance.log()
        return loss

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

class IsoMaxPlusLossSecondPart(torch.nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale = 10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale
    
    def forward(self, logits, targets):
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        epsilon = 1e-10
        # Calcolo delle distanze
        distances = -logits
        # Applica softmax lungo la dimensione delle classi (dim=1)
        probabilities_for_training = torch.nn.Softmax(dim=1)(-self.entropic_scale * distances)
        # Reshape dei tensori per applicare gather
        batch_size, num_classes, height, width = probabilities_for_training.shape
        probabilities_for_training = probabilities_for_training.view(batch_size, num_classes, -1)  # [B, C, H*W]
        targets = targets.view(batch_size, -1)  # [B, H*W]
        # Seleziona le probabilità corrispondenti ai target pixel per pixel
        probabilities_at_targets = probabilities_for_training.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H*W]
        # Calcola la loss come media della log-loss
        loss = -torch.log(probabilities_at_targets + epsilon).mean()
        return loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        """
        alpha: bilanciamento del contributo tra classi (usato per bilanciare classi sbilanciate)
        gamma: fattore che riduce l'importanza degli esempi ben classificati
        weight: pesi opzionali per le classi
        reduction: 'mean', 'sum' o 'none' per definire come aggregare la loss
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] - output grezzo del modello
        targets: [B, H, W] - etichette vere con valori tra 0 e num_classes-1
        """
        # Calcola la cross-entropy senza riduzione
        ce_loss = self.cross_entropy(logits, targets)  # [B, H, W]
        # Calcola la probabilità predetta per la classe corretta
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        probs_at_targets = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]
        # Applica il fattore di focalizzazione (1 - p_t)^gamma
        focal_factor = (1.0 - probs_at_targets) ** self.gamma
        # Calcola la Focal Loss
        focal_loss = self.alpha * focal_factor * ce_loss
        # Riduci la loss in base al parametro reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

def train_model(model, train_loader, val_loader, optimizer, criterion1, criterion2, num_epochs, encoder):
    epoch_losses=[]
    for epoch in range(num_epochs):
        print("----- TRAINING - EPOCH", epoch, "-----")
        model.train()
        epoch_loss = 0
        for images, labels in tqdm(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            inputs = Variable(images)
            targets = Variable(labels)
            # Forward pass
            logits = model(inputs, only_encode=encoder)
            # Calcolo delle loss
            loss1 = criterion1(logits, targets[:, 0])
            loss2 = criterion2(logits, targets[:, 0])
            loss = loss1 + loss2
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        average_epoch_loss_train = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_epoch_loss_train}")
        epoch_losses.append(average_epoch_loss_train)
        
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()
                inputs = Variable(images)
                targets = Variable(labels)
                # Forward pass
                outputs = model(inputs, only_encode=encoder)
                # Calcolo delle loss
                loss1 = criterion1(outputs, targets[:, 0])
                loss2 = criterion2(outputs, targets[:, 0])
                loss = loss1 + loss2
                # calculate iou
                preds = outputs.argmax(dim=1)
                total_iou += calculate_iou(preds, targets, NUM_CLASSES).mean().item()
        print(f"Validation Mean IoU: {total_iou / len(val_loader)}")
    if encoder == False:
        epochs = list(range(1, num_epochs + 1))  # Epoche da 1 a N
        plt.plot(epochs, epoch_losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Andamento della Loss')
        plt.legend()
        plt.savefig('Loss.png')

def main():
    datadir = "../datasets"
    num_workers = 4
    batch_size = 4
    height = 512
    num_epochs = 50
    model_file = importlib.import_module("erfnet")
    model = model_file.ERFNet(NUM_CLASSES)
    model.cuda()
    '''
    model = torch.nn.DataParallel(model).cuda()
    print("========== ENCODER TRAINING ===========")
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.3653597831726	
    weight[1] = 4.4237880706787	
    weight[2] = 2.9691488742828	
    weight[3] = 5.3442072868347	
    weight[4] = 5.2983593940735	
    weight[5] = 5.2275490760803	
    weight[6] = 5.4394111633301	
    weight[7] = 5.3659925460815	
    weight[8] = 3.4170460700989	
    weight[9] = 5.2414722442627	
    weight[10] = 4.7376127243042	
    weight[11] = 5.2286224365234	
    weight[12] = 5.455126285553	
    weight[13] = 4.3019247055054	
    weight[14] = 5.4264230728149	
    weight[15] = 5.4331531524658	
    weight[16] = 5.433765411377	
    weight[17] = 5.4631009101868	
    weight[18] = 5.3947434425354
    weight[19] = 0
    weight = weight.cuda()
    # Trasformazione per input + Dataset + Dataloader
    co_transform = MyCoTransform(True, augment=True, height=height)
    co_transform_val = MyCoTransform(True, augment=False, height=height)
    dataset_train = cityscapes(datadir, co_transform, 'train')
    dataset_val = cityscapes(datadir, co_transform_val, 'val')
    train_loader = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    # Ottimizzatore e criterio di perdita
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
    #criterion1 = CrossEntropyLoss2d(weight)
    criterion1 = FocalLoss(weight=weight)
    criterion2 = IsoMaxPlusLossSecondPart()
    # Esegui il training
    train_model(model, train_loader, val_loader, optimizer, criterion1, criterion2, num_epochs, True)
    '''
    print("========== DECODER TRAINING ===========")
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
    #carico l'encoder
    '''
    pretrainedEnc = next(model.children()).encoder
    model = model_file.ERFNet(NUM_CLASSES, encoder=pretrainedEnc) 
    model = torch.nn.DataParallel(model).cuda()
    '''
    # Ottimizzatore e criterio di perdita
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
    #criterion1 = CrossEntropyLoss2d(weight)
    criterion1 = FocalLoss(weight=weight)
    criterion2 = IsoMaxPlusLossSecondPart()
    # Esegui il training
    train_model(model, train_loader, val_loader, optimizer, criterion1, criterion2, num_epochs, False)
    #save the model
    torch.save(model.state_dict(), "erfnet_cityscapes.pth")

if __name__ == '__main__':
    main()


#python main3.py