import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from dataset import cityscapes
from erfnet import ERFNet
from temperature_scaling import ModelWithTemperature

NUM_CLASSES = 20
IMAGE_SIZE = (512, 512)  # Dimensione dell'immagine dopo il resize

input_transform_cityscapes = Compose([
    Resize((512, 512), Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize((512, 512), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # Ignora etichette con valore 255
])

def main(args):
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model:", modelpath)
    print("Loading weights:", weightspath)

    model = ERFNet(NUM_CLASSES)

    # Carica i pesi
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, "not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")

    if not args.cpu:
        model = model.cuda()

    model.eval()

    # Carica il dataset di validazione
   loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Applica il Temperature Scaling
    print("Applying Temperature Scaling...")
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(loader)

    print(f"Optimal temperature: {scaled_model.temperature.item()}")

    # Salva il modello scalato
    torch.save(scaled_model.state_dict(), '/kaggle/working/scaled_model.pth')
    print("Scaled model saved successfully.")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadDir', default="/kaggle/input/pretrained_model_erfnet/pytorch/default/")
    parser.add_argument('--loadWeights', default="1model_best.pth.tar")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/kaggle/input/cityscapes-correctlabels/Cityscape")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--input_transform', default=None)  # Adatta le trasformazioni
    parser.add_argument('--target_transform', default=None)

    main(parser.parse_args())
