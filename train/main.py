# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

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



import os
import torch
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from my_transforms import MyCoTransform  # Assumendo che sia definita
from my_dataset import cityscapes  # Assumendo che sia definita
from my_metrics import iouEval  # Assumendo che sia definita
from my_loss import CrossEntropyLoss2d  # Assumendo che sia definita

def train(args, model, enc=False):
    # Inizializzazione
    best_acc = 0
    NUM_CLASSES = 20  # Modifica con il numero corretto di classi
    weight = torch.ones(NUM_CLASSES)
    savedir = f'../save/{args.savedir}'
    os.makedirs(savedir, exist_ok=True)
    
    # Configura pesi per il loss
    try:
        if enc:
            weight[:19] = torch.tensor([
                2.3654, 4.4238, 2.9691, 5.3442, 5.2984, 5.2275, 5.4394, 5.3660, 
                3.4170, 5.2415, 4.7376, 5.2286, 5.4551, 4.3019, 5.4264, 5.4332, 
                5.4338, 5.4631, 5.3947
            ])
        else:
            weight[:19] = torch.tensor([
                2.8149, 6.9850, 3.7890, 9.9428, 9.7702, 9.5111, 10.3114, 10.0265, 
                4.6323, 9.5608, 7.8698, 9.5169, 10.3737, 6.6616, 10.2605, 10.2879, 
                10.2898, 10.4054, 10.1381
            ])
    except IndexError as e:
        print("Error setting weights:", e)
    
    # Controlla il dataset
    assert os.path.exists(args.datadir), f"Error: dataset directory '{args.datadir}' not found"
    co_transform = MyCoTransform(enc, augment=True, height=args.height)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)

    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    
    # Verifica dataset
    assert len(dataset_train) > 0, "Training dataset is empty!"
    assert len(dataset_val) > 0, "Validation dataset is empty!"

    # DataLoader
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # Configurazione CUDA
    if args.cuda:
        weight = weight.cuda()
        model = model.cuda()
    
    criterion = CrossEntropyLoss2d(weight)
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1 - (epoch / args.num_epochs)), 0.9))
    start_epoch = 1

    # Carica checkpoint se necessario
    if args.resume:
        checkpoint_path = os.path.join(savedir, 'checkpoint_enc.pth.tar' if enc else 'checkpoint.pth.tar')
        assert os.path.exists(checkpoint_path), f"Checkpoint not found at '{checkpoint_path}'"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Checkpoint loaded from epoch {start_epoch}")

    # Inizio training
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_loss = []
        print(f"----- TRAINING - EPOCH {epoch} -----")

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images, only_encode=enc)
            loss = criterion(outputs, labels[:, 0])
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # Epoch summary
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch} training loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = []
            for images, labels in loader_val:
                if args.cuda:
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images, only_encode=enc)
                val_loss.append(criterion(outputs, labels[:, 0]).item())
            
            avg_val_loss = sum(val_loss) / len(val_loss)
            print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, os.path.join(savedir, 'checkpoint.pth.tar'))
        print(f"Checkpoint saved for epoch {epoch}")
    
    return model


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    print("model path is: ", args.model)
    print("Current working directory:", os.getcwd())
    print("Looking for model file at:", os.path.abspath(args.model + ".py"))
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
        else:
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    
    print("Training dataset path:", args.datadir)
    print("Files in training directory:", os.listdir(os.path.join(args.datadir, "train")))

    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
