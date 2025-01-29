import os
from PIL import Image
import numpy as np

# Transformamos las imágenes ahora a 8 bits para subirlas a la WebApp 

# Ruta a la carpeta donde están las imágenes originales
carpeta_imagenes = r"C:\Users\34711\Desktop\xraygpt\test_V4"
# Nueva carpeta donde se guardarán las imágenes convertidas
carpeta_8bits = r"C:\Users\34711\Desktop\xraygpt\tests_8bits"

# Crear la carpeta 'train_8bits' si no existe
if not os.path.exists(carpeta_8bits):
    os.makedirs(carpeta_8bits)

# Función para convertir la imagen a 8 bits (RGB) y guardarla en la nueva carpeta
def convertir_a_8_bits(imagen_path, imagen_nombre):
    try:
        # Abrir la imagen
        img = Image.open(imagen_path)
        img_array = np.array(img)  # Convertir la imagen a array numpy
        
        # Verificar si la imagen tiene más de 8 bits de profundidad
        if img_array.max() > 255:
            print(f"Imagen {imagen_nombre} tiene más de 8 bits de profundidad, normalizándola...")
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            img_array = img_array.astype(np.uint8)  # Convertir a 8 bits

            # Convertir el array de vuelta a imagen
            img = Image.fromarray(img_array)

        # Guardar la imagen convertida en la nueva carpeta
        nueva_ruta = os.path.join(carpeta_8bits, imagen_nombre)
        img.save(nueva_ruta)
        print(f"Imagen convertida y guardada: {nueva_ruta}")
    except Exception as e:
        print(f"Error al procesar {imagen_path}: {e}")

# Recorrer todas las imágenes en la carpeta
for archivo in os.listdir(carpeta_imagenes):
    ruta_imagen = os.path.join(carpeta_imagenes, archivo)
    
    # Verificar que el archivo sea una imagen con extensión válida
    if archivo.lower().endswith(('png', 'jpg', 'jpeg')):
        convertir_a_8_bits(ruta_imagen, archivo)

print("Proceso de conversión completado.")

