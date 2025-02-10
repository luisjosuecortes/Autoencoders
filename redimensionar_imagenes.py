import os
from PIL import Image
import shutil

def crear_directorios_salida():
    """
    Crea la estructura de directorios para las imágenes redimensionadas.
    Elimina el directorio si ya existe para evitar mezclar con resultados anteriores.
    """
    directorio_base = 'imagenes_512'
    if os.path.exists(directorio_base):
        shutil.rmtree(directorio_base)
    
    print(f"Creando directorios en: {os.path.abspath(directorio_base)}")
    os.makedirs(directorio_base)
    os.makedirs(os.path.join(directorio_base, 'RealArt'))
    os.makedirs(os.path.join(directorio_base, 'AiArtData'))
    return directorio_base

def redimensionar_imagen(ruta_entrada, ruta_salida, tamano=(512, 512)):
    """
    Redimensiona una imagen al tamaño deseado, estirando la imagen para llenar
    todo el espacio sin bordes.
    
    Args:
        ruta_entrada: Ruta de la imagen original
        ruta_salida: Ruta donde se guardará la imagen redimensionada
        tamano: Tupla con el ancho y alto deseados
    """
    try:
        print(f"Procesando imagen: {ruta_entrada}")
        with Image.open(ruta_entrada) as img:
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar directamente al tamaño objetivo usando LANCZOS
            # para mejor calidad en la reducción
            img_redimensionada = img.resize(tamano, Image.Resampling.LANCZOS)
            
            # Ajustar el contraste ligeramente para mantener la imagen viva
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img_redimensionada)
            img_final = enhancer.enhance(1.1)
            
            # Guardar con alta calidad
            img_final.save(ruta_salida, 'JPEG', quality=95)
            print(f"✓ Imagen guardada en: {ruta_salida}")
    except Exception as e:
        print(f"❌ Error procesando {ruta_entrada}: {str(e)}")

def procesar_directorio(directorio_entrada, directorio_salida):
    """
    Procesa todas las imágenes en un directorio.
    
    Args:
        directorio_entrada: Ruta del directorio con las imágenes originales
        directorio_salida: Ruta donde se guardarán las imágenes procesadas
    """
    print(f"\nProcesando directorio: {directorio_entrada}")
    
    if not os.path.exists(directorio_entrada):
        print(f"❌ Error: El directorio {directorio_entrada} no existe")
        return
    
    archivos = os.listdir(directorio_entrada)
    imagenes = [f for f in archivos if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not imagenes:
        print(f"⚠️ No se encontraron imágenes en {directorio_entrada}")
        return
    
    print(f"Encontradas {len(imagenes)} imágenes para procesar")
    
    for nombre_archivo in imagenes:
        ruta_entrada = os.path.join(directorio_entrada, nombre_archivo)
        ruta_salida = os.path.join(directorio_salida, 
                                  os.path.splitext(nombre_archivo)[0] + '.jpg')
        redimensionar_imagen(ruta_entrada, ruta_salida)

def main():
    """
    Función principal que coordina el proceso de redimensionamiento
    """
    print("🔄 Iniciando proceso de redimensionamiento de imágenes...")
    directorio_base = crear_directorios_salida()
    
    # Procesar carpeta de arte real
    procesar_directorio('imagenes/RealArt/RealArt', 
                       os.path.join(directorio_base, 'RealArt'))
    
    # Procesar carpeta de arte de IA
    procesar_directorio('imagenes/AiArtData/AiArtData', 
                       os.path.join(directorio_base, 'AiArtData'))
    
    print("\n✨ Proceso completado!")
    print(f"📁 Las imágenes redimensionadas están en: {os.path.abspath(directorio_base)}")

if __name__ == "__main__":
    main() 