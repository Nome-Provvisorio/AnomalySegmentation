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


def main():
    # Configurazioni principali
    model_path = "/kaggle/input/pretrained_model_erfnet/pytorch/default/erfnet.py"
    weights_path = "/kaggle/input/pretrained_model_erfnet/pytorch/default/1model_best.pth.tar"
    data_dir = "/kaggle/input/cityscapes-correctlabels/Cityscape"
    batch_size = 8
    num_workers = 4
    use_cuda = torch.cuda.is_available()

    # Carica il modello
    model = load_model(model_path, weights_path, NUM_CLASSES, use_cuda)

    # Crea il DataLoader per il set di validazione
    valid_dataset = cityscapes(data_dir, input_transform_cityscapes, target_transform_cityscapes, subset='val')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Applica il Temperature Scaling
    scaled_model = apply_temperature_scaling(model, valid_loader)

    # Salva il modello scalato
    torch.save(scaled_model.state_dict(), '/kaggle/working/scaled_model.pth')
    print("Modello scalato salvato con successo!")


if __name__ == '__main__':
    main()
