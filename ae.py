from copyreg import constructor
from importlib_metadata import pass_none
import numpy as np
import os
from os.path import join
from PIL import Image
import time
import matplotlib.pyplot as plt
import enum
import csv

from tensorflow.keras.layers import Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import inception_v3

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import DeepFool
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import NewtonFool


class Attacks(enum.Enum):
  FGSM = 'FGSM'
  PGD = 'PGD'
  DF = 'DF'
  CW = 'CW'
  BIM = 'BIM'
  NF = 'NF'
  
class AE:
  def __init__(self, params, classifier):
    self.params = params
    self.module = inception_v3
    self.classifier = classifier

    prefix = params['prefix']
    epsilon = params['epsilon']
    eps_step = params['eps_step']
    max_iter = params['max_iter']
    attack_type = params['attack_type']
    epsilon_str = params['epsilon_str']
    self.params['dataset'] = f"{prefix}/imagenette2-320"
    self.params['org_dataset_images'] = f"{prefix}/Original/Images"
    self.params['org_dataset_npz'] = f"{prefix}/Original/NPZ"
    self.params['adv_dataset_images'] = f"{prefix}/{attack_type}/{epsilon_str}/Images"
    self.params['adv_dataset_npz'] = f"{prefix}/{attack_type}/{epsilon_str}/NPZ"
    self.params['results_image'] = f"{prefix}/Results/{attack_type}_{epsilon_str}.jpg"
    self.params['results_csv'] = f"{prefix}/Metadata/{attack_type}_{epsilon_str}.csv"

    attacks = dict()
    for attack in Attacks:
      if attack.name == 'FGSM':
        attacks[attack.name] = FastGradientMethod(estimator=self.classifier, eps=epsilon, eps_step=eps_step)
      if attack.name == 'PGD':
        attacks[attack.name] = ProjectedGradientDescent(estimator=self.classifier, eps=epsilon, eps_step=eps_step)
      if attack.name == 'DF':
        attacks[attack.name] = DeepFool(classifier=self.classifier, epsilon=epsilon, max_iter=max_iter)
      if attack.name == 'CW':
        attacks[attack.name] = CarliniL2Method(classifier=self.classifier, max_iter=max_iter)
      if attack.name == 'BIM':
        attacks[attack.name] = BasicIterativeMethod(self.classifier, epsilon, eps_step)
      if attack.name == 'NF':
        attacks[attack.name] = NewtonFool(self.classifier, max_iter=max_iter)
    
    self.attack = attacks[attack_type]
  
  ############################ PREPROCESSING DATA ##############################
  def to_image(self, image):
    OldMin, OldMax = np.min(image), np.max(image)
    NewMin, NewMax = 0.0, 256.0
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)
    return ((((image - OldMin) * NewRange) / OldRange) + NewMin).astype(np.uint8)
  ##############################################################################

  ############################# GENERATING DATA ################################
  def create_org_dataset(self):
    image_size = self.params['image_size']
    dataset = self.params['dataset']
    org_dataset_images = self.params['org_dataset_images']
    org_dataset_npz = self.params['org_dataset_npz']
    per_class = self.params['per_class']
    class_to_index = self.parmas['class_to_index']
    
    X_processed = []
    X = []
    y = []

    # Generating Original Dataset
    classes_dir = sorted(os.listdir(join(dataset, 'train')))
    for ind, dir in enumerate(classes_dir):

      class_dir = join(dataset, 'train', dir)
      print("Started class " + dir, end='\r')
      
      processed_images = []
      cnt = 0
      for image in sorted(os.listdir(class_dir)):
      
        # Opening the image and resizing it
        image = Image.open(join(class_dir, image)).resize((image_size[0], image_size[1]))
        image_array = np.asarray(image)
      
        # There are some images which are grey scaled, this avoids that
        if image_array.shape != image_size:
          continue
        
        # Saving the images
        file_name = join(org_dataset_images, 'original_' + str(dir) + '_' + str(cnt) + '.png')
        image.save(file_name)

        X.append(image_array)
        
        # Processing for npz data 
        image_array = self.module.preprocess_input(image_array)
        processed_images.append(image_array)

        X_processed.append(image_array)
        y.append(int(class_to_index[dir]))
        cnt+=1
        if cnt == per_class:
          break 

      processed_images = np.array(processed_images)

      # Saving the processed images in npz form
      file_name = join(org_dataset_npz, f'original_{dir}.npz')
      np.savez_compressed(file_name, data=processed_images)

      print("Done class " + dir, end='\n') 
    
    
    X_processed = np.array(X_processed)
    X = np.array(X)
    y = np.array(y)

    return {'X': X, 'X_processed': X_processed, 'y': y}

  def create_adv_dataset(self, X_all, classes):
    per_class = self.params['per_class']
    attack = self.attack
    adv_dataset_images = self.params['adv_dataset_images']
    adv_dataset_npz = self.params['adv_dataset_npz']
    class_to_index = self.params['class_to_index']
    class_list = self.params['class_list']
    
    adv_generating_time = 0
    
    for class_name in classes:
      # Finding indexes
      class_index = class_list.index(class_name)
      start_index = class_index * per_class
      end_index = (class_index+1) * per_class

      # Attacking on subset of dataset
      start = time.perf_counter()

      X_adv = attack.generate(x=X_all[start_index : end_index], verbose=True)

      end = time.perf_counter()
      adv_generating_time += round(end-start,2)

      y_adv = np.argmax(self.classifier.predict(X_adv), axis=1)
      org_accuracy = (np.sum(class_to_index[class_name] == y_adv))/len(y_adv)
      print('Adversarial Accuracy :', org_accuracy)
      
      # Saving images
      cnt = 0
      for ind, image_array in enumerate(X_adv):
        # converting to 0 - 255 range
        image = Image.fromarray(self.to_image(image_array))
        
        # file path
        file_name = f'adversarial_{class_name}_{cnt}.png'
        file_path = join(adv_dataset_images , file_name)
        
        # saving file
        image.save(file_path)
        
        cnt += 1

      # Saving npz

      # file path
      file_name = f'adversarial_{class_name}.npz'
      file_path = join(adv_dataset_npz, file_name)
      
      # saving file
      np.savez_compressed(file_path, data=X_adv)
      
    self.params['adv_generating_time'] = adv_generating_time
  ##############################################################################

  ############################## LOADING DATA ##################################
  def get_org_dataset(self, fromImage=True):
    per_class = self.params['per_class']
    org_dataset_images = self.params['org_dataset_images']
    org_dataset_npz = self.params['org_dataset_npz']
    class_to_index = self.params['class_to_index']
    class_list = self.params['class_list']
    
    X = []
    y = []
    
    if fromImage:
      for class_name in class_list:
        for ind in range(per_class):
          # File path
          image_file_name = f'original_{class_name}_{ind}.png'
          image_file_path = join(org_dataset_images, image_file_name)

          # Opening
          image = Image.open(image_file_path)
          image_array = np.asarray(image)
          X.append(image_array)
        
          # Extracting Class
          class_name = image_file_name.split('_')[1]
          y.append(int(class_to_index[class_name]))
      
    else:
      for class_name in class_list:
        # File path
        image_file_name = f'original_{class_name}.npz'
        image_file_path = join(org_dataset_npz, image_file_name)

        # Opening
        processed_images = np.load(image_file_path)['data']
        for processed_image in processed_images[:per_class]:
          X.append(processed_image)

        # Extracting Class
        for times in range(per_class):
          y.append(int(class_to_index[class_name]))

    X=np.array(X)
    y=np.array(y)
    
    if fromImage:
      X = self.module.preprocess_input(X)
      
    return X,y

  def get_adv_dataset(self, fromImage=True):
    per_class = self.params['per_class']
    adv_dataset_images = self.params['adv_dataset_images']
    adv_dataset_npz = self.params['adv_dataset_npz']
    class_list = self.params['class_list']
    
    X_adv = []

    if fromImage:
      for class_name in class_list:
        for ind in range(per_class):
          # File path
          file_name = f'adversarial_{class_name}_{ind}.png'
          file_path =  join(adv_dataset_images, file_name)
          
          # Opening image
          image = Image.open(file_path)
          image = np.asarray(image)
          X_adv.append(image)

    else:
      for class_name in self.class_list:
        # File path
        file_name = f'adversarial_{class_name}.npz'
        file_path = join(adv_dataset_npz, file_name)

        # Opening images
        processed_images = np.load(file_path)['data']
        for processed_image in processed_images[:per_class]:
          X_adv.append(processed_image)

    X_adv=np.array(X_adv)
    if fromImage:
      X_adv = self.module.preprocess_input(X_adv)
      
    return X_adv
  ##############################################################################

  ############################# DATA VISULIZATION ##############################
  def show_images(self, X:np.ndarray, y:np.ndarray, X_adv=None, y_adv=None, n=5):
    # Choosing random indexes
    inds = np.random.choice(X.shape[0], n, replace=False)

    if X is None or X_adv is None:
      fig, axs = plt.subplots(1, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[i].imshow(self.to_image(X[ind]))
        axs[i].set_title(y[ind])

    else:
      fig, axs = plt.subplots(3, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[0, i].imshow(self.to_image(X[ind]))
        axs[0, i].set_title(y[ind])    

        # Adversarial Image
        axs[1, i].imshow(self.to_image(X_adv[ind]))
        axs[1, i].set_title(y_adv[ind])    

        # Difference Image
        axs[2, i].imshow(abs(self.to_image(X[ind]) - self.to_image(X_adv[ind])))
        axs[2, i].set_title("Difference")      
  ##############################################################################
  
  ################################## RESULTS ###################################
  def get_predictions(self, X):
    return np.argmax(self.classifier.predict(X), axis=1)
  
  def get_accuracy(self, X, org_class):
    y = np.argmax(self.classifier.predict(X), axis=1)
    org_accuracy = (np.sum(org_class == y))/len(y)
    return org_accuracy
  
  def save(self, X:np.ndarray, y:np.ndarray, X_adv=None, y_adv=None, n=5):
    params = self.params
    results_image = self.params['results_image']
    results_csv = self.params['results_csv']
    
    # Choosing random indexes
    inds = np.random.choice(X.shape[0], n, replace=False)

    if X is None or X_adv is None:
      fig, axs = plt.subplots(1, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[i].imshow(self.to_image(X[ind]))
        axs[i].set_title(y[ind])

    else:
      fig, axs = plt.subplots(3, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[0, i].imshow(self.to_image(X[ind]))
        axs[0, i].set_title(y[ind])    

        # Adversarial Image
        axs[1, i].imshow(self.to_image(X_adv[ind]))
        axs[1, i].set_title(y_adv[ind])    

        # Difference Image
        axs[2, i].imshow(abs(self.to_image(X[ind]) - self.to_image(X_adv[ind])))
        axs[2, i].set_title("Difference")      

    plt.savefig(results_image)
    
    header = ['attack_type', 'image_size', 'epsilon', 'org_accuracy', 'adv_accuracy', 'time']
    if os.path.isfile(results_csv) == False:
      with open(results_csv, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
    data = [params['attack_type'], params['image_size'], params['epsilon'], params['org_accuracy'], params['adv_accuracy'], params['adv_generating_time']]
    with open(results_csv, 'a') as f:
      writer = csv.writer(f)
      writer.writerow(data)
  ##############################################################################

  