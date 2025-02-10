# Detector de Imágenes Generadas por IA usando Autoencoders

Este proyecto implementa un detector de imágenes generadas por IA utilizando autoencoders convolucionales profundos con bloques residuales. El sistema está diseñado para aprender las características intrínsecas de las imágenes reales y usar el error de reconstrucción como métrica para la detección.

## Características

- Autoencoder convolucional profundo con bloques residuales
- Normalización por lotes para estabilidad en el entrenamiento
- Métricas de evaluación completas (F1-Score, AUC-ROC)
- Visualizaciones de resultados
- Análisis detallado de errores de reconstrucción

## Requisitos

```bash
Pillow==10.2.0
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
matplotlib==3.7.1
tqdm==4.65.0
```

## Estructura del Proyecto

```
.
├── autoencoder_detector.py   # Implementación principal del autoencoder
├── evaluar_detector.py       # Script de evaluación
├── redimensionar_imagenes.py # Utilidad para preprocesamiento
├── paper_autoencoder.tex     # Documentación técnica
└── requirements.txt          # Dependencias del proyecto
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/luisjosuecortes/Autoencoders.git
cd Autoencoders
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Preparar las imágenes:
```bash
python redimensionar_imagenes.py
```

2. Entrenar el modelo:
```bash
python autoencoder_detector.py
```

3. Evaluar el modelo:
```bash
python evaluar_detector.py
```

## Resultados

El modelo actual alcanza:
- F1-Score: 0.71
- AUC-ROC: 0.52
- Umbral óptimo: 0.001021

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de crear un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. 