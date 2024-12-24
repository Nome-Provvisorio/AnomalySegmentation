import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from temperature_scaling import ModelWithTemperature
from erfnet import ERFNet
from dataset import cityscapes  # Assicurati che il modulo sia disponibile
from transform import Relabel, ToLabel

from PIL import Image


# Configurazioni globali
NUM_CLASSES = 20
IMAGE_SIZE = (512, 512)  # Dimensione dell'immagine dopo il resize

# Trasformazioni per il dataset
input_transform_cityscapes = Compose([
    Resize(IMAGE_SIZE, Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize(IMAGE_SIZE, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),  # Ignora etichetta 255 e sostituiscila con 19
])


def load_model(model_path, weights_path, num_classes, use_cuda=True):
    """
    Carica il modello ERFNet con i pesi salvati.
    """
    print(f"Caricamento del modello da: {model_path}")
    print(f"Caricamento dei pesi da: {weights_path}")

    model = ERFNet(num_classes)

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

    state_dict = torch.load(weights_path, map_location='cpu')
    model = load_my_state_dict(model, state_dict)

    if use_cuda:
        model = model.cuda()

    model.eval()
    print("Modello caricato con successo!")
    return model


def apply_temperature_scaling(model, valid_loader):
    """
    Applica il Temperature Scaling al modello.
    """
    print("Applying Temperature Scaling...")
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(valid_loader)
    print(f"Temperatura ottimale: {scaled_model.temperature.item()}")
    return scaled_model


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
