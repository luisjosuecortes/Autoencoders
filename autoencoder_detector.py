import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import seaborn as sns

class ImagenesArteDataset(Dataset):
    """Dataset personalizado para las imágenes de arte."""
    def __init__(self, directorio_raiz, transform=None):
        """
        Args:
            directorio_raiz: Directorio que contiene las imágenes
            transform: Transformaciones opcionales a aplicar
        """
        self.directorio_raiz = directorio_raiz
        self.transform = transform
        self.imagenes_paths = []
        
        # Buscar todas las imágenes en el directorio
        for archivo in os.listdir(directorio_raiz):
            if archivo.lower().endswith(('.jpg', '.jpeg')):
                self.imagenes_paths.append(os.path.join(directorio_raiz, archivo))
                
        print(f"Encontradas {len(self.imagenes_paths)} imágenes en {directorio_raiz}")

    def __len__(self):
        return len(self.imagenes_paths)

    def __getitem__(self, idx):
        imagen_path = self.imagenes_paths[idx]
        imagen = Image.open(imagen_path)
        
        if self.transform:
            imagen = self.transform(imagen)
            
        return imagen

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(in_channels) 
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x))) # Formula = 
        out = self.bn2(self.conv2(out))
        out += residual 
        out = self.relu(out)
        return out

class Autoencoder(nn.Module):
    """Autoencoder para la detección de imágenes generadas por IA."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 512x512 -> 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128 -> 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 256x256 -> 512x512
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def entrenar_autoencoder(modelo, train_loader, num_epochs, device):
    """Entrena el autoencoder."""
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)
    historial_perdida = []
    
    modelo.to(device)
    print(f"Entrenando en: {device}")
    
    for epoca in range(num_epochs):
        perdida_total = 0
        num_batches = 0
        
        barra_progreso = tqdm(train_loader, desc=f'Época {epoca+1}/{num_epochs}')
        for batch in barra_progreso:
            imagenes = batch.to(device)
            
            reconstruidas = modelo(imagenes)
            perdida = criterio(reconstruidas, imagenes)
            
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            
            perdida_total += perdida.item()
            num_batches += 1
            barra_progreso.set_postfix({'loss': perdida.item()})
        
        perdida_promedio = perdida_total / num_batches
        historial_perdida.append(perdida_promedio)
        print(f'Época [{epoca+1}/{num_epochs}], Pérdida: {perdida_promedio:.6f}')
    
    return historial_perdida

def main():
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Definir transformaciones
    transform = transforms.Compose([ #Transformaciones a aplicar a las imágenes
        transforms.ToTensor(),
    ])
    
    # Crear dataset y dataloader
    dataset = ImagenesArteDataset( #Carga las imágenes en el directorio
        directorio_raiz='imagenes_512/RealArt',
        transform=transform
    )
    
    train_loader = DataLoader( #Carga los datos en batches de 32 imágenes
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Crear y entrenar modelo
    modelo = Autoencoder() 
    num_epochs = 50
    historial_perdida = entrenar_autoencoder(modelo, train_loader, num_epochs, device)
    
    # Guardar modelo
    torch.save(modelo.state_dict(), 'modelo_autoencoder.pth')
    
    # Graficar pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(historial_perdida)
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.savefig('perdida_entrenamiento.png')
    plt.close()

if __name__ == "__main__":
    main() 