import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from autoencoder_detector import Autoencoder, ImagenesArteDataset
import os
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import seaborn as sns

def calcular_error_reconstruccion(modelo, imagen, device):
    """Calcula el error de reconstrucción para una imagen."""
    modelo.eval()
    with torch.no_grad():
        imagen = imagen.to(device)
        reconstruida = modelo(imagen.unsqueeze(0))
        error = torch.nn.functional.mse_loss(reconstruida.squeeze(0), imagen)
        return error.item()

def evaluar_conjunto_imagenes(modelo, directorio, device, transform):
    """Evalúa un conjunto de imágenes y calcula sus errores de reconstrucción."""
    dataset = ImagenesArteDataset(directorio, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    errores = []
    paths = []
    
    print(f"\nEvaluando imágenes en: {directorio}")
    for i, imagen in enumerate(loader):
        error = calcular_error_reconstruccion(modelo, imagen.squeeze(0), device)
        errores.append(error)
        paths.append(dataset.imagenes_paths[i])
        
        if i % 10 == 0: 
            print(f"Procesadas {i+1}/{len(loader)} imágenes")
    
    return errores, paths

def visualizar_distribucion_errores(errores_reales, errores_ia):
    """Visualiza la distribución de errores para ambos tipos de imágenes."""
    plt.figure(figsize=(12, 6))
    
    plt.hist(errores_reales, bins=50, alpha=0.5, label='Arte Real', color='blue')
    plt.hist(errores_ia, bins=50, alpha=0.5, label='Arte IA', color='red')
    
    plt.xlabel('Error de Reconstrucción')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores de Reconstrucción')
    plt.legend()
    plt.savefig('distribucion_errores.png')
    plt.close()

def calcular_metricas(errores_reales, errores_ia, umbral):
    """Calcula todas las métricas de evaluación."""
    # Preparar etiquetas y predicciones
    y_true = np.concatenate([np.zeros(len(errores_reales)), np.ones(len(errores_ia))])
    y_scores = np.concatenate([errores_reales, errores_ia])
    y_pred = (y_scores > umbral).astype(int)
    
    # Calcular curva ROC y AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calcular F1-Score
    f1 = f1_score(y_true, y_pred)
    
    # Calcular y visualizar matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Visualizar curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return roc_auc, f1

def encontrar_umbral_optimo(errores_reales, errores_ia):
    """Encuentra el umbral óptimo para clasificación."""
    todos_errores = np.concatenate([errores_reales, errores_ia])
    umbrales_candidatos = np.linspace(min(todos_errores), max(todos_errores), 100)
    
    mejor_umbral = 0
    mejor_f1 = 0
    
    y_true = np.concatenate([np.zeros(len(errores_reales)), np.ones(len(errores_ia))])
    
    for umbral in umbrales_candidatos:
        y_pred = (np.concatenate([errores_reales, errores_ia]) > umbral).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral
    
    return mejor_umbral, mejor_f1

def main():
    # Configurar dispositivo y cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder()
    modelo.load_state_dict(torch.load('modelo_autoencoder.pth'))
    modelo.to(device)
    modelo.eval()
    
    # Definir transformaciones
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Evaluar imágenes
    errores_reales, paths_reales = evaluar_conjunto_imagenes(
        modelo, 'imagenes_512/RealArt', device, transform
    )
    
    errores_ia, paths_ia = evaluar_conjunto_imagenes(
        modelo, 'imagenes_512/AiArtData', device, transform
    )
    
    # Visualizar distribución de errores
    visualizar_distribucion_errores(errores_reales, errores_ia)
    
    # Encontrar umbral óptimo y calcular métricas
    umbral, f1 = encontrar_umbral_optimo(errores_reales, errores_ia)
    roc_auc, f1_final = calcular_metricas(errores_reales, errores_ia, umbral)
    
    print(f"\nResultados finales:")
    print(f"Umbral óptimo: {umbral:.6f}")
    print(f"F1-Score: {f1_final:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")
    
    # Guardar resultados detallados
    with open('resultados_detallados.txt', 'w') as f:
        f.write(f"Métricas globales:\n")
        f.write(f"Umbral óptimo: {umbral:.6f}\n")
        f.write(f"F1-Score: {f1_final:.2f}\n")
        f.write(f"AUC-ROC: {roc_auc:.2f}\n\n")
        
        f.write("Resultados para imágenes reales:\n")
        for path, error in zip(paths_reales, errores_reales):
            f.write(f"{path}: {error:.6f}\n")
        
        f.write("\nResultados para imágenes generadas por IA:\n")
        for path, error in zip(paths_ia, errores_ia):
            f.write(f"{path}: {error:.6f}\n")

if __name__ == "__main__":
    main() 