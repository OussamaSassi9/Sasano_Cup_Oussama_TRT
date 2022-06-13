from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image


def aug(img):
    datagen = ImageDataGenerator(        
            rotation_range=30, fill_mode='nearest')
    channels = img.shape[-1] if img.ndim == 3 else 1

    print(channels)

    if img.ndim==2:
        x=img.reshape((1, )+img.shape+(1,))
    else:
        x=img.reshape((1, )+img.shape)

    
    i = 0
    for batch in datagen.flow(x, batch_size=16,
                            save_to_dir= r'dataset\FLUSH\train\augmented',
                            save_prefix='flush',
                            save_format='bmp'):    
        i += 1    
        if i > 10:        
            break




image_directory = r'dataset\FLUSH\train\augmented'
list_of_images = os.listdir(image_directory)
for image in list_of_images:
    img = io.imread(os.path.join(image_directory, image))
    aug(img)