import pydicom
import numpy as np
from PIL import Image
import cv2
import os
from joblib import Parallel, delayed

# El preprocesado de los archivos DICOM se realiza a parte ya que la carpeta train original ocupa 20 GB...
#... así que descargo el .zip con 'train' y 'test' y ejecuto el preprocesado en local. 

def resize_with_padding(img, desired_size=(256, 256)):
    old_size = img.shape[:2]  # old_size is in (height, width) format

    # Determine whether to scale up or scale down
    is_upscaling = old_size[0] < desired_size[0] or old_size[1] < desired_size[1]

    # Choose interpolation method based on whether you are upscaling or downscaling
    if is_upscaling:
        interp_method = cv2.INTER_CUBIC  # Better for enlarging
    else:
        interp_method = cv2.INTER_AREA   # Better for reducing size

    # Calculate new size to maintain aspect ratio
    ratio = float(desired_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image
    img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=interp_method)

    # Calculate padding to reach the desired size
    delta_w = desired_size[1] - new_size[1]
    delta_h = desired_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # Color of padding, change if necessary

    # Add padding to the resized image
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_img

def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return (img - img_min) / (img_max - img_min)

def convert_dicom_to_png(input_path, output_path):
    ds = pydicom.dcmread(input_path)
    img = ds.pixel_array.astype(float)

    # Apply windowing if metadata available
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        img = window_image(img, ds.WindowCenter, ds.WindowWidth)

    # Normalize the image
    img /= img.max()

    # Convert Monochrome1 to Monochrome2
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        img = 1 - img

    # Special processing for images with Bits Allocated = 8 and Photometric Interpretation = MONOCHROME2
    if ds.BitsAllocated == 8 and ds.PhotometricInterpretation == 'MONOCHROME2':
        img = resize_with_padding(img)

        # Apply Gaussian filter to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Convert to 8-bit PNG
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode='L')
    else:
        # Resize and padding
        img = resize_with_padding(img)

        # Convert to 16-bit PNG
        img = (img * 65535).astype(np.uint16)
        img_pil = Image.fromarray(img, mode='I;16')

    img_pil.save(output_path)

def process_images(input_folder, output_folder):
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.dcm')]
    output_files = [os.path.join(output_folder, f.replace('.dcm', '.png')) for f in os.listdir(input_folder) if f.endswith('.dcm')]
    existing_files = {os.path.basename(f) for f in os.listdir(output_folder)}

    files_to_process = [(input_files[i], output_files[i]) for i in range(len(input_files)) if os.path.basename(output_files[i]) not in existing_files]

    Parallel(n_jobs=-1)(delayed(process_single_image)(f[0], f[1]) for f in files_to_process)

def process_single_image(input_path, output_path):
    try:
        convert_dicom_to_png(input_path, output_path)
        print(f"Imagen procesada y guardada: {output_path}")
    except Exception as e:
        print(f"Error al procesar {input_path}: {e}")


# input_folder = ## Insertar ruta con .dicom                      lo mismo para train y para test
# output_folder = ## Insertar ruta con donde se guardarán en
# os.makedirs(output_folder, exist_ok=True)  
# process_images(input_folder, output_folder)

