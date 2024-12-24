import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from temperature_scaling import ModelWithTemperature
from erfnet import ERFNet
from dataset import cityscapes
from transform import Relabel, ToLabel
from PIL import Image

# Configurazioni globali
NUM_CLASSES = 20
IMAGE_SIZE = (512, 512)  # Dimensione immagine dopo il resize

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


def main(args):
    # Carica il modello
    model = load_model(args.model_path, args.weights_path, NUM_CLASSES, args.cuda)

    # Crea il DataLoader per il set di validazione
    valid_dataset = cityscapes(args.data_dir, input_transform_cityscapes, target_transform_cityscapes, subset='val')
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Applica il Temperature Scaling
    scaled_model = apply_temperature_scaling(model, valid_loader)

    # Salva il modello scalato
    torch.save(scaled_model.state_dict(), args.output_path)
    print(f"Modello scalato salvato con successo in: {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temperature Scaling for ERFNet with Cityscapes Dataset")

    # Aggiungi argomenti configurabili
    parser.add_argument('--model_path', type=str, required=True, help='Path al file del modello (es. erfnet.py)')
    parser.add_argument('--weights_path', type=str, required=True, help='Path ai pesi del modello (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory del dataset Cityscapes')
    parser.add_argument('--output_path', type=str, default='scaled_model.pth', help='Path per salvare il modello scalato')
    parser.add_argument('--batch_size', type=int, default=8, help='Dimensione del batch per il DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Numero di worker per il DataLoader')
    parser.add_argument('--cuda', action='store_true', help='Usa CUDA se disponibile')

    args = parser.parse_args()
    main(args)
