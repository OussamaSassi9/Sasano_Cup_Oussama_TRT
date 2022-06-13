from utils import *
from main_process import Main_Process
from glob import glob
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import os
import shutil

class MakeDataset(Main_Process):
    
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.main_process = Main_Process(images_folder)

    def dataset_annotation(self, img, anomalies, image_path, train_test):
        image_label = image_path.split('\\')[-1].split('.')[0]
        if train_test == 1:
            train_test_split = 'train'
        else:
            train_test_split = 'test'
        
        n = len(anomalies)  
        for i, detected_anomaly in enumerate(anomalies):
            min_X,max_X,min_Y,max_Y= calculate_min_max(detected_anomaly)
            #cv2.rectangle(img, (min_X-10, min_Y-10), (max_X+10, max_Y+10), (255,255,255), 3)
            #rect = cv2.minAreaRect(cluster)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            area = cv2.contourArea(detected_anomaly)
            if(area > 50):
                #crop the anomaly
                anomaly_img = img[min_Y-10:max_Y+10, min_X-10:max_X+10]
                
                #show the anomaly
                cv2.imshow('Detected Anomaly {}/{}'.format(i, n), anomaly_img)
                cv2.waitKey(0)
                
                #annotate the anomaly
                annotation = int(input('1: FLUSH --- 2: OTHER ANOMALY\n'))
                
                #write the anomaly to the train/test folder
                if annotation == 1:
                    cv2.imwrite(os.path.join('dataset/FLUSH', train_test_split, image_label+'_'+str(i) +'.jpg'), anomaly_img)
              
                else:
                    cv2.imwrite(os.path.join('dataset/OTHER' , train_test_split, image_label+'_'+str(i)+'.jpg'), anomaly_img)
                    
                
    def exc_pipeline(self):
        #set directories for the anomalies extraction (dataset genereateion)
        purge = ''
        if os.path.isdir('dataset'):
            while purge != 'y' and purge != 'n': 
                purge = input('Do you want to purge the existing dataset: [Y/N]:').lower()
        if purge == 'y':
                shutil.rmtree('dataset')
        os.makedirs(os.path.join('dataset', 'FLUSH', 'train'), exist_ok=True)
        os.makedirs(os.path.join('dataset', 'FLUSH', 'test'), exist_ok=True)
    
        os.makedirs(os.path.join('dataset', 'OTHER', 'train'), exist_ok=True)
        os.makedirs(os.path.join('dataset', 'OTHER', 'test'), exist_ok=True)
        
        for image_path in glob(self.images_folder+'/*.bmp'):
            img = cv2.imread(image_path)
            croped = image_cropping(img)

            img_bin = self.main_process.image_binarization(croped)
            contours = self.main_process.contours_detection(img_bin)
            anomalies = self.main_process.regroup_contours_with_DBScan(contours)
            
            #ask if this image is in train or test
            print('Current image: ', image_path)
            train_test = int(input('1: Train \n0: TEST\n'))
            #annotate
            self.dataset_annotation(croped, anomalies, image_path, train_test)
            

if __name__ == '__main__':
    images_folder = 'images'
    dataset_generator = MakeDataset(images_folder)
    dataset_generator.exc_pipeline()
    