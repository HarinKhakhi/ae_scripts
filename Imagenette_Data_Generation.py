########################## IMPORTING LIBRARIES ##########################
import numpy as np
import pickle
from urllib.request import urlopen
import time
import csv 

import os
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import inception_v3

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import DeepFool
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import NewtonFool
##############################################################################

################################# PARAMETERS #################################
epsilon = 0.01
eps_step = epsilon / 10

attack_type = 'DF' # 'FGSM', 'PGD', 'DF', 'CW', 'BIM', 'NF'
##############################################################################

################################# CONSTANTS ##################################
image_size = (300, 300, 3)
image_size_str = str(image_size[0]) + '_' + str(image_size[1]) + '_' + str(image_size[2])
epsilon_str = str(epsilon).split('.')[0] + "_" + str(epsilon).split('.')[1]
perClass = 100

prefix = "/home/admin-pc/Desktop/ae"

dataset = f"{prefix}/imagenette2-320"

org_dataset_images = f"{prefix}/Original/Images"
org_dataset_npz = f"{prefix}/Original/NPZ"

adv_dataset_images = f"{prefix}/{attack_type}/{epsilon_str}/Images"
adv_dataset_npz = f"{prefix}/{attack_type}/{epsilon_str}/NPZ"

results_image = f"{prefix}/Results/{attack_type}_{epsilon_str}.jpg"
results_csv = f"{prefix}/Metadata/{attack_type}_{epsilon_str}.csv"

for directory in [org_dataset_images, org_dataset_npz, adv_dataset_images, adv_dataset_npz]:
  if not os.path.isdir(directory):
    print('created ', directory)
    os.makedirs(directory)
    
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
index_to_class = {class_index: class_name for class_name, class_index in class_to_index.items()} # inversting the dictionary
class_list = sorted(list(class_to_index.keys()))

index_to_name = pickle.load(urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )

nb_classes = 1000 # Number of ImageNet classes

module = inception_v3
model = inception_v3.InceptionV3( input_tensor=Input(shape=image_size),
                                  include_top=True, 
                                  weights='imagenet', 
                                  classifier_activation='softmax')
loss = CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model,
                                    nb_classes=nb_classes,
                                    loss_object=loss,
                                    input_shape=image_size,
                                    clip_values=(0,1))

attacks = {
    'FGSM': FastGradientMethod(estimator=classifier, eps=epsilon, eps_step=eps_step),
    'PGD': ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=eps_step),
    'DF': DeepFool(classifier=classifier, epsilon=epsilon, max_iter=10),
    'CW': CarliniL2Method(classifier=classifier, max_iter=10),
    'BIM': BasicIterativeMethod(classifier, epsilon, eps_step),
    'NF': NewtonFool(classifier, max_iter=10)
}

assert attack_type in attacks.keys()

attack = attacks[attack_type]
##############################################################################

############################ PREPROCESSING DATA ##############################
def to_image(image):
  OldMin, OldMax = np.min(image), np.max(image)
  NewMin, NewMax = 0.0, 256.0
  OldRange = (OldMax - OldMin)  
  NewRange = (NewMax - NewMin)
  return ((((image - OldMin) * NewRange) / OldRange) + NewMin).astype(np.uint8)
##############################################################################

############################# GENERATING DATA ################################
def create_org_dataset():
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
      image_array = module.preprocess_input(image_array)
      processed_images.append(image_array)

      X_processed.append(image_array)
      y.append(int(class_to_index[dir]))
      cnt+=1
      if cnt == perClass:
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

def create_adv_dataset(X, y):
  global adv_generating_time
  start = time.perf_counter()
  
  X_adv = attack.generate(x=X, verbose=True)
  
  end = time.perf_counter()
  adv_generating_time = round(end-start,2)
  
  # Saving images
  cnt = 0
  for ind, image_array in enumerate(X_adv):
    # converting to 0 - 255 range
    image = Image.fromarray(to_image(image_array))
    
    # file path
    file_name = 'adversarial_' + index_to_class[y[ind]] + '_' + str(cnt) + '.png'
    file_path = join(adv_dataset_images , file_name)
    
    # saving file
    image.save(file_path)
    
    cnt += 1
    cnt %= perClass


  # Saving npz
  for i, start_index in enumerate(range(0, perClass*len(class_list), perClass)):
    # file path
    file_name = f'adversarial_{class_list[i]}.npz'
    file_path = join(adv_dataset_npz, file_name)
    
    # saving file
    np.savez_compressed(file_path, data=X_adv[start_index:start_index+perClass])
##############################################################################

############################## LOADING DATA ##################################
def get_org_dataset(fromImage=True):
  X = []
  y = []
  
  if fromImage:
    for class_name in class_list:
      for ind in range(perClass):
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
      for processed_image in processed_images[:perClass]:
        X.append(processed_image)

      # Extracting Class
      for times in range(perClass):
        y.append(int(class_to_index[class_name]))

  X=np.array(X)
  y=np.array(y)
  return X,y

def get_adv_dataset(fromImage=True):
  
  X_adv = []

  if fromImage:
    for class_name in class_list:
      for ind in range(perClass):
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
      for processed_image in processed_images[:perClass]:
        X_adv.append(processed_image)

  X_adv=np.array(X_adv)
  return X_adv
##############################################################################

############################# DATA VISULIZATION ##############################
def show_images(X:np.ndarray, y:np.ndarray, X_adv=None, y_adv=None, n=5):
  
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
      axs[0, i].set_title(y[ind])    

      # Adversarial Image
      axs[1, i].imshow(to_image(X_adv[ind]))
      axs[1, i].set_title(y_adv[ind])    

      # Difference Image
      axs[2, i].imshow(abs(to_image(X[ind]) - to_image(X_adv[ind])))
      axs[2, i].set_title("Difference")      

  plt.savefig(results_image)
##############################################################################

############################# MAIN SCRIPT ####################################
# Generating Original Data
if (len(os.listdir(org_dataset_images)) == 0) and (len(os.listdir(org_dataset_npz)) == 0):
  print('Creating Original Data')
  data = create_org_dataset()
  print('Generated Original Data')
  del data

# Loading Data
fromImage = False
X, org_class = get_org_dataset(fromImage)
if fromImage:
  X = module.preprocess_input(X)

# Calculating accuracy
y = np.argmax(classifier.predict(X), axis=1)
org_accuracy = (np.sum(org_class == y))/len(y)
print('Original Accuracy :', org_accuracy)

# Generating Adversarial Data
print('Generating Adversarial Data')
create_adv_dataset(X, org_class)
print('Generated Adversarial Data')

# Loading Adversarial Data
fromImage=False
X_adv = get_adv_dataset(fromImage)
if fromImage:
  X_adv = module.preprocess_input(X_adv)

y_adv = np.argmax(classifier.predict(X_adv), axis=1)
adv_accuracy = (np.sum(org_class == y_adv))/len(y_adv)
print('Adversarial Accuracy :', adv_accuracy)

show_images(X=X, y=org_class, X_adv=X_adv, y_adv=y_adv)
##############################################################################
header = ['attack_type', 'image_size', 'epsilon', 'org_accuracy', 'adv_accuracy', 'time']
if os.path.isfile(results_csv) == False:
  with open(results_csv, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
data = [attack_type, image_size, epsilon, org_accuracy, adv_accuracy, adv_generating_time]
with open(results_csv, 'a') as f:
  writer = csv.writer(f)
  writer.writerow(data)
  
print('\033[92m')
print('Attack: ', attack_type)
print('Image Size: ', image_size)
print('Epsilon: ', epsilon)
print('Original Accuracy: ', org_accuracy)
print('Adversarial Accuracy: ', adv_accuracy)
print('Time Taken: ', adv_generating_time)
print('\033[0m')