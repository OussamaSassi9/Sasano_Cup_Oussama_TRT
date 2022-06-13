import os
import numpy as np
import shutil
import random

# creating train / val /test
root_dir = 'data/'
new_root = 'AllDataset/'
classes = ['hgs_flush', 'hgs_not_flush']

for cls in classes:
    os.makedirs(root_dir + new_root+ cls + '/train/'  ,exist_ok=True)
    os.makedirs(root_dir +new_root +  cls + '/test/' ,exist_ok=True)
    
## creating partition of the data after shuffeling

for cls in classes:
    src = root_dir + cls # folder to copy images from
    print(src)

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    ## here 0.75 = training ratio , (0.95-0.75) = validation ratio , (1-0.95) = training ratio  
    train_FileNames,val_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.75),int(len(allFileNames)*0.95)])

    # #Converting file names from array to list

    train_FileNames = [src+'/'+ name for name in train_FileNames]
    test_FileNames = [src+'/' + name for name in test_FileNames]

  
    
    ## Copy pasting images to target directory

    for name in train_FileNames:
        shutil.copy(name, root_dir + new_root+cls+'/train/' )



    for name in test_FileNames:
        shutil.copy(name,root_dir + new_root+cls+'/test/' )

