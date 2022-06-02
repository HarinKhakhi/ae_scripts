from operator import contains
import numpy as np
import os
from os.path import join
from PIL import Image
import time
import matplotlib.pyplot as plt
import enum
import csv
import cv2

from tensorflow.keras.layers import Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import inception_v3

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import DeepFool
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import CarliniLInfMethod
from art.attacks.evasion import NewtonFool

from art.estimators.classification import TensorFlowV2Classifier

class Attacks(enum.Enum):
  FGSM = 'FGSM'
  PGD = 'PGD'
  DF = 'DF'
  CW = 'CW'
  CW_L2 = 'CW'
  CW_LInf = 'CW_LInf'
  BIM = 'BIM'
  NF = 'NF'

# Example of parameters
# 
# epsilon = 0.1
# eps_step = epsilon/10
# attack_type = Attacks.FGSM.name
# targeted_attack = False
# prefix = '/content/drive/MyDrive/AE_Resources/Imagenette/Datasets'
# image_size = (300, 300, 3)
# per_class = 100
# max_iter = 100

def initialize(params):
  image_size = params['image_size']
  prefix = params['prefix']
  epsilon = params['epsilon']
  eps_step = params['eps_step']
  max_iter = params['max_iter']
  attack_type = params['attack_type']
  targeted_attack = params['targeted_attack']

  class_to_index = {
    'n01440764': 0,
    'n02102040': 217,
    'n02979186': 482,
    'n03000684': 491,
    'n03028079': 497,
    'n03394916': 566,
    'n03417042': 569,
    'n03425413': 571,
    'n03445777': 574,
    'n03888257': 701,
  }
  class_list = sorted(list(class_to_index.keys()))
  epsilon_str = str(epsilon).split('.')[0] + "_" + str(epsilon).split('.')[1]
  module = inception_v3
  model = inception_v3.InceptionV3( input_tensor=Input(shape=image_size),
                                    include_top=True, 
                                    weights='imagenet', 
                                    classifier_activation='softmax')
  loss = CategoricalCrossentropy(from_logits=False)
  classifier = TensorFlowV2Classifier(model=model,
                                      nb_classes=1000,
                                      loss_object=loss,
                                      input_shape=image_size,
                                      clip_values=(0,1))
  
  attack = None

  if attack_type == 'FGSM':
    attack = FastGradientMethod(estimator=classifier, eps=epsilon, eps_step=eps_step, targeted=bool(targeted_attack))
  elif attack_type == 'PGD':
    attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=eps_step, targeted=bool(targeted_attack))
  elif attack_type == 'DF':
    attack = DeepFool(classifier=classifier, epsilon=epsilon, max_iter=max_iter)
  elif attack_type == 'CW' :
    attack = CarliniL2Method(classifier=classifier, max_iter=max_iter, targeted=bool(targeted_attack))
  elif attack_type == 'CW_LInf':
    attack = CarliniLInfMethod(classifier=classifier, max_iter=max_iter, targeted=bool(targeted_attack))
  elif attack_type == 'BIM':
    attack = BasicIterativeMethod(classifier, epsilon, eps_step, targeted=bool(targeted_attack))
  elif attack_type == 'NF':
    attack = NewtonFool(classifier, max_iter=max_iter)
  else:
    raise NameError("Attack is not available")
  
  params['module'] = module
  params['classifier'] = classifier
  params['attack'] = attack
  
  params['class_to_index'] = class_to_index
  params['class_list'] = class_list
  params['epsilon_str'] = epsilon_str

  params['dataset'] = f"{prefix}/Datasets/imagenette2-320"
  params['org_dataset_images'] = f"{prefix}/Datasets/Original/Images"
  params['org_dataset_npz'] = f"{prefix}/Datasets/Original/NPZ"
  params['adv_dataset_images'] = f"{prefix}/Datasets/{attack_type}/{epsilon_str}/Images"
  params['adv_dataset_npz'] = f"{prefix}/Datasets/{attack_type}/{epsilon_str}/NPZ"
  params['autoencoders_dir'] = f"{prefix}/Autoencoders"
  params['results_img_dir'] = f"{prefix}/Results"
  params['results_meta_dir'] = f"{prefix}/Metadata"
  params['results_image'] = f"{prefix}/Results/{attack_type}_{epsilon_str}.jpg"
  params['results_csv'] = f"{prefix}/Metadata/{attack_type}_{epsilon_str}.csv"
  
  # Datasets directory
  for directory in [params['org_dataset_images'], params['org_dataset_npz'], params['adv_dataset_images'], params['adv_dataset_npz']]:
    if not os.path.isdir(directory):
      os.makedirs(directory)
      print(f'{directory} created')

  # Results directory
  for directory in [params['autoencoders_dir'], params['results_img_dir'], params['results_meta_dir']]:
    if not os.path.isdir(directory):
      os.makedirs(directory)
      print(f'{directory} created')

############################## PROCESSING DATA ##############################
def to_image(image):
  OldMin, OldMax = 0, 1
  NewMin, NewMax = 0.0, 256.0
  OldRange = (OldMax - OldMin)  
  NewRange = (NewMax - NewMin)
  return ((((image - OldMin) * NewRange) / OldRange) + NewMin).astype(np.uint8)

# def set_preprocessor(params, func):
#   params['preprocessor'] = func

def preprocess(params, X):
  return cv2.normalize(X, 
                None, 
                alpha = 0, beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)
  # return params['preprocessor'](X)

def batch_data(X_all, y_all, per_batch=2):
  X_batched, y_batched = [], []
  for batch_ind in range(int(X_all.shape[0] / (per_batch*10))):
    per_class = int(X_all.shape[0]/10)
    
    for class_ind in range(10):
      class_start_index = class_ind * per_class
      for image in X_all[(class_start_index + (batch_ind*per_batch)) : (class_start_index + ((batch_ind+1)*per_batch))]:
        X_batched.append(image)
        y_batched.append(class_ind)
        
  X_batched = np.array(X_batched)
  y_batched = np.array(y_batched)
  
  return X_batched, y_batched
def split_data(params, X, y, train_size=0.8):
  per_class = params['per_class']
  X_train, X_test = [], []
  y_train, y_test = [], []

  # total images per class reduces
  newPerClass_train = int(per_class * train_size)
  newPerClass_test = per_class - newPerClass_train

  # split every class
  for class_start_ind in range(0, X.shape[0], per_class):
    for ind in range(per_class):
      if ind < newPerClass_train:
        X_train.append(X[class_start_ind + ind])
        y_train.append(y[class_start_ind + ind])
      else:
        X_test.append(X[class_start_ind + ind])
        y_test.append(y[class_start_ind + ind])

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
     
  return (X_train, X_test, y_train, y_test)
##############################################################################

############################# GENERATING DATA ################################
def create_org_dataset(params):
  image_size = params['image_size']
  dataset = params['dataset']
  org_dataset_images = params['org_dataset_images']
  org_dataset_npz = params['org_dataset_npz']
  per_class = params['per_class']
  class_to_index = params['class_to_index']
  
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
      image_array = preprocess(params, image_array)
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

def create_adv_dataset(params, X_all, classes, y_all=None, saveImage=True):
  per_class = params['per_class']
  attack = params['attack']
  adv_dataset_images = params['adv_dataset_images']
  adv_dataset_npz = params['adv_dataset_npz']
  class_to_index = params['class_to_index']
  class_list = params['class_list']
  classifier = params['classifier']
  
  X_adv_all = []
  adv_generating_time = 0
  adv_accuracy = []
  for class_name in classes:
    # Finding indexes
    class_index = class_list.index(class_name)
    start_index = class_index * per_class
    end_index = (class_index+1) * per_class

    print(f'Attacking Class {class_index}: {class_name}')
    # Attacking on subset of dataset
    start = time.perf_counter()

    X_adv = attack.generate(x=X_all[start_index : end_index], y=y_all, verbose=True)

    end = time.perf_counter()
    adv_generating_time += round(end-start,2)

    y_adv = np.argmax(classifier.predict(X_adv), axis=1)
    adv_accuracy.append(np.sum(class_to_index[class_name] == y_adv)/len(y_adv))
    print('Current Adversarial Accuracy :', sum(adv_accuracy)/len(adv_accuracy))
    
    if saveImage:
      # Saving images
      cnt = 0
      for ind, image_array in enumerate(X_adv):
        # converting to 0 - 255 range
        image = Image.fromarray(to_image(image_array))
        
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
    else :
      for attacked_image in X_adv:
        X_adv_all.append(attacked_image)
          
  params['adv_generating_time'] = adv_generating_time
  
  return np.array(X_adv_all)
##############################################################################

############################## LOADING DATA ##################################
def get_org_dataset(params, fromImage=True):
  per_class = params['per_class']
  org_dataset_images = params['org_dataset_images']
  org_dataset_npz = params['org_dataset_npz']
  class_to_index = params['class_to_index']
  class_list = params['class_list']
  
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
    X = preprocess(params, X)

  return X,y

def get_adv_dataset(params, fromImage=True):
  per_class = params['per_class']
  adv_dataset_images = params['adv_dataset_images']
  adv_dataset_npz = params['adv_dataset_npz']
  class_list = params['class_list']
  module = params['module']
  
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
    for class_name in class_list:
      # File path
      file_name = f'adversarial_{class_name}.npz'
      file_path = join(adv_dataset_npz, file_name)

      # Opening images
      processed_images = np.load(file_path)['data']
      for processed_image in processed_images[:per_class]:
        X_adv.append(processed_image)

  X_adv=np.array(X_adv)
  if fromImage:
    X_adv = preprocess(params, X_adv)
    
  return X_adv
##############################################################################

############################# DATA VISULIZATION ##############################
def show_images(X=None, y=None, X_adv=None, y_adv=None, n=5, batch=False, title=''):
  np.random.seed(1234)

  if batch:
    # Choosing random indexes
    batch_ind = np.random.choice(X.shape[0], 1, replace=False)[0]
    inds = np.random.choice(X.shape[1], n, replace=False)

    if X is None or X_adv is None:
      fig, axs = plt.subplots(1, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[i].imshow(to_image(X[batch_ind][ind]))
        axs[i].set_title(y[batch_ind][ind])

    else:
      fig, axs = plt.subplots(3, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[0, i].imshow(to_image(X[batch_ind][ind]))
        axs[0, i].set_title(y[batch_ind][ind])    

        # Adversarial Image
        axs[1, i].imshow(to_image(X_adv[batch_ind][ind]))
        axs[1, i].set_title(y_adv[batch_ind][ind])    

        # Difference Image
        axs[2, i].imshow(abs(to_image(X[batch_ind][ind]) - to_image(X_adv[batch_ind][ind])))
        axs[2, i].set_title("Difference")   
  else:
    # Choosing random indexes
    inds = np.random.choice(X.shape[0], n, replace=False)

    if X is None or X_adv is None:
      fig, axs = plt.subplots(1, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[i].imshow(to_image(X[ind]))
        axs[i].set_title(y[ind])

    else:
      fig, axs = plt.subplots(3, n, figsize=(3*n,10))
      # Plotting data 
      for i, ind in enumerate(inds):

        # Original Image
        axs[0, i].imshow(to_image(X[ind]))
        axs[0, i].set_title(f"Org: {y[ind]}")    

        # Adversarial Image
        axs[1, i].imshow(to_image(X_adv[ind]))
        axs[1, i].set_title(f"Cleaned/Adv: {y_adv[ind]}")    

        # Difference Image
        axs[2, i].imshow(abs(to_image(X[ind]) - to_image(X_adv[ind])))
        axs[2, i].set_title("Difference") 
  
  plt.suptitle(title)
##############################################################################

def set_autoencoder_name(params, model_name):
  cnt = 0
  found = False
  while not found:
    new_model_name = f'{model_name}_{cnt}'
    for dir in os.listdir(params['autoencoders_dir']):
      if new_model_name == dir:
        found = True
        cnt+=1
        break
    if not found: 
      break
  
  params['model_name'] = f'{model_name}_{cnt}'
  return params['model_name']
############################# AUTOENCODER ##############################
def train_autoencoder(params, autoencoder, X, X_adv): 
  for X_image, X_adv_image in zip(X, X_adv):
    # Checking if data is batched or not
    if not len(X.shape) == 5:
      # expanding the dimention to merge both images into a array
      X_image=np.expand_dims(X_image, axis=0)
      X_adv_image=np.expand_dims(X_adv_image, axis=0)
      
    # Mapping original -> original & adversarial -> original. 
    autoencoder.fit(np.vstack([X_image, X_adv_image]),
                    np.vstack([X_image, X_image]), epochs=10)

  autoencoder.save(join(params['autoencoders_dir'], params['model_name']))
##############################################################################

def test_autoencoder(params, autoencoder, X, X_adv, y):
  
  if(len(X.shape)==5):
    X_pred, X_adv_pred = None, None
    for batch in range(X.shape[0]):
      if X_pred is None:
        X_pred = autoencoder.predict(X[batch])
      else:
        X_pred = np.append(X_pred, autoencoder.predict(X[batch]), axis=1)
      if X_adv_pred is None:
        X_adv_pred = autoencoder.predict(X_adv[batch])
      else:
        X_adv_pred = np.append(X_adv_pred, autoencoder.predict(X_adv[batch]), axis=1)
  else:  
    X_pred = autoencoder.predict(X)
    X_adv_pred = autoencoder.predict(X_adv)
    
  y_pred = get_predictions(params, X_pred)
  y_adv_pred = get_predictions(params, X_adv_pred)
  
  params['org_accuracy'] = get_accuracy(y, get_predictions(X))
  params['adv_accuracy'] = get_accuracy(y, get_predictions(X_adv))
  
  params['org_cleaned_accuracy'] = get_accuracy(y, y_pred)
  params['adv_cleaned_accuracy'] = get_accuracy(y, y_adv_pred)
  
  print('Original Non-Cleaned Classifier Accuracy:', params['org_accuracy'])
  print('Adversarial Non-Cleaned Classifier Accuracy:', params['adv_accuracy']) 
  print('Original Cleaned Classifier Accuracy:', params['org_cleaned_accuracy'])
  print('Adversarial Cleaned Classifier Accuracy:', params['adv_cleaned_accuracy'])

################################## RESULTS ###################################
def get_predictions(params, X):
  return np.argmax(params['classifier'].predict(X), axis=1)

def get_accuracy(y, y_pred):
  accuracy = (np.sum(y_pred == y))/len(y)
  return accuracy

def save(params, X=None, y=None, X_adv=None, y_adv=None, n=5, batch=False, title=''):
  results_image = params['results_image']
  results_csv = params['results_csv']
  
  show_images(X, y, X_adv, y_adv, n, batch, title)
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