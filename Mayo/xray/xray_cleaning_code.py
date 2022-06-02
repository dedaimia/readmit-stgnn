#!/usr/bin/env python3
import os
from skimage import exposure
from skimage.io import imread, imsave
import pandas as pd
import sys
import numpy as np
def equalization_plus_stretch(image, clip, percentile_lo, percentile_hi):

    img_adaptive = exposure.equalize_adapthist(image)#, clip_limit=1)#clip)
    p_lo, p_hi = np.percentile(img_adaptive, (2, 99))#(percentile_lo, percentile_hi))
    img_rescale = exposure.rescale_intensity(img_adaptive, in_range=(p_lo, p_hi))

    global img_rescale_interactive, image_name_global
    img_rescale_interactive = img_rescale
    return img_rescale

header = '/mnt/storage/Readmission/colab_readmission/xrays_all_48h/'
files = os.listdir(header+'Images/')
df = pd.DataFrame()
for i in range(len(files)):
    print(i, files[i])
    image_path = header+'Images/'+files[i]
    corrected_image_path = header+'Images_processed/'+files[i]
    try:
        im = imread(image_path)

        im = equalization_plus_stretch(image =im, clip=(.005,.2,.005), percentile_lo=(1,100,.5), percentile_hi=(1,100,.5))
   
        h = np.histogram(im.flatten()/im.max(), bins=2)
    
    
        suspicious = True
        if (h[0][1]<1.2*h[0][0]) or (h[0][0]<1.2*h[0][1]):
            suspicious = False
        
        if h[0][1]<1.1*h[0][0]:
            im = 1-im
            print('invert', h[0][0], h[0][1])
            
        imsave(corrected_image_path, im)
        df.at[i, 'image_path'] = image_path
        df.at[i, 'corrected_image_path'] = corrected_image_path
        df.at[i, 'suspicious'] = suspicious
    except:
        print('not readable')
    sys.stdout.flush()
df.to_csv('/mnt/storage/Readmission/colab_readmission/xrays_all_48h/xrays_processing_results.csv')
print('done')
sys.stdout.flush()
