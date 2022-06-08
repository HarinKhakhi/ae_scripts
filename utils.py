from enum import Enum
import os 
from os.path import join
import time

from tensorflow.keras.layers import Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import inception_v3, resnet50

import numpy as np
from PIL import Image
from io import BytesIO
from cv2 import CV_16U, normalize, NORM_MINMAX, CV_32F
import matplotlib.pyplot as plt

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import CarliniLInfMethod
from art.estimators.classification import TensorFlowV2Classifier, KerasClassifier
from art.preprocessing.preprocessing import Preprocessor

class Attacks(Enum):
  FGSM = 'FGSM'
  PGD = 'PGD'
  CW_L0 = 'CW_L0'
  CW_L2 = 'CW_L2'
  CW_LInf = 'CW_LInf'
  BIM = 'BIM'


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
class_list = list(class_to_index.keys())

def preprocess(image):
  return normalize(image, 
          None, 
          alpha = 0, beta = 1, 
          norm_type = NORM_MINMAX, 
          dtype = CV_32F)

def to_image(image):
  return normalize(image, 
          None, 
          alpha = 0, beta = 255, 
          norm_type = NORM_MINMAX, 
          dtype = CV_16U)

def get(default, value):
  if value is None:
    return default
  else:
    return value

def create_org_dataset(source_path, target_path, **kwargs):
  org_dataset_images = target_path + '/Images'
  org_dataset_npz = target_path + '/NPZ'
  image_size = get((300,300,3), kwargs.get('image_size'))
  per_class = get(100, kwargs.get('per_class'))
  
  # Creating the directories 
  os.mkdirs(org_dataset_images)
  os.mkdirs(org_dataset_npz)
    
  # Initialization  
  X_processed = []
  X = []
  y = []

  # Generating Original Dataset
  classes_dir = sorted(os.listdir(join(source_path, 'train')))
  for directory in classes_dir:

    class_dir = join(source_path, 'train', directory)
    print("Started class " + directory, end='\r')
    
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
      file_name = join(org_dataset_images, 'original_' + str(directory) + '_' + str(cnt) + '.png')
      image.save(file_name)

      X.append(image_array)
      
      # Processing for npz data 
      image_array = preprocess(image_array)
      processed_images.append(image_array)

      X_processed.append(image_array)
      y.append(int(class_to_index[directory]))
      cnt+=1
      if cnt == per_class:
        break 

    processed_images = np.array(processed_images)

    # Saving the processed images in npz form
    file_name = join(org_dataset_npz, f'original_{directory}.npz')
    np.savez_compressed(file_name, data=processed_images)

    print("Done class " + directory, end='\n') 
  
  # Casting arrays to np array
  X_processed = np.array(X_processed)
  X = np.array(X)
  y = np.array(y)

  return {'X': X, 'X_processed': X_processed, 'y': y}

def get_attack(attack_type, classifier, **kwargs):
  attack = None
  epsilon = get(0.01, kwargs.get('epsilon'))
  eps_step = epsilon/10
  max_iter = get(100, kwargs.get('max_iter'))
  targeted = get(False, kwargs.get('targeted'))
  
  if attack_type == 'FGSM':
    attack = FastGradientMethod(estimator=classifier, eps=epsilon, eps_step=eps_step, targeted=targeted)
  elif attack_type == 'PGD':
    attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=eps_step, targeted=targeted)
  elif attack_type == 'CW' :
    attack = CarliniL2Method(classifier=classifier, max_iter=max_iter, targeted=targeted)
  elif attack_type == 'CW_LInf':
    attack = CarliniLInfMethod(classifier=classifier, max_iter=max_iter, targeted=targeted)
  elif attack_type == 'BIM':
    attack = BasicIterativeMethod(classifier, epsilon, eps_step, targeted=targeted)
  else:
    raise NameError("Attack is not available")
  
  return attack

def create_adv_dataset(classifier, attack_type, X, y=None, **kwargs):
  per_class = get(100, kwargs.get('per_class'))
  classes_to_attack = get(class_list, kwargs.get('class_list'))
  attack = get_attack(classifier=classifier, attack_type=attack_type, **kwargs)
  
  X_adv_all = []
  adv_generating_time = 0
  adv_accuracy = [] 
  for class_name in classes_to_attack:
    # Finding indexes
    class_index = class_list.index(class_name)
    start_index = class_index * per_class
    end_index = (class_index+1) * per_class

    print(f'Attacking Class {class_index}: {class_name}')
    # Attacking on subset of dataset
    start = time.perf_counter()

    X_adv = attack.generate(x=X[start_index : end_index], y=y[start_index : end_index], verbose=get(True, kwargs.get('verbose')))

    end = time.perf_counter()
    adv_generating_time += round(end-start,2)

    y_adv = np.argmax(classifier.predict(X_adv), axis=1)
    adv_accuracy.append(np.sum(class_to_index[class_name] == y_adv)/len(y_adv))
    print('Current Adversarial Accuracy :', sum(adv_accuracy)/len(adv_accuracy))
    
    if kwargs.get('save_image'):
      adv_dataset_images = kwargs.get('target_path') + '/Images'
      adv_dataset_npz = kwargs.get('target_path') + '/NPZ'
      
      # Creating the directories 
      os.mkdirs(adv_dataset_images)
      os.mkdirs(adv_dataset_npz)
      
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
      file_name = f'adversarial_{class_name}.npz'
      file_path = join(adv_dataset_npz, file_name)
      np.savez_compressed(file_path, data=X_adv)
    else :
      for attacked_image in X_adv:
        X_adv_all.append(attacked_image)
          
  print('time taken: ', adv_generating_time)
  return np.array(X_adv_all)

def load_dataset(source_path, kind, **kwargs):
  
  per_class = get(100,kwargs.get('per_class'))
  source_images =  source_path + '/Images'
  source_npz = source_path + '/NPZ'
  
  X = []
  y = []
  
  if kwargs.get('from_image'):
    for class_name in class_list:
      for ind in range(per_class):
        # File path
        image_file_name = f'{original}_{class_name}_{ind}.png'
        image_file_path = join(source_images, image_file_name)

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
      image_file_name = f'{kind}_{class_name}.npz'
      image_file_path = join(source_npz, image_file_name)

      # Opening
      processed_images = np.load(image_file_path)['data']
      for processed_image in processed_images[:per_class]:
        X.append(processed_image)

      # Extracting Class
      for times in range(per_class):
        y.append(int(class_to_index[class_name]))

  X=np.array(X)
  y=np.array(y)
  
  return X,y

def get_model(name, **kwargs):
  image_size=(300,300,3)
  if name == 'inception':
    model = inception_v3.InceptionV3(input_tensor=Input(shape=get(image_size, kwargs.get('image_size'))),weights='imagenet')
    loss = CategoricalCrossentropy(from_logits=False)
    classifier = TensorFlowV2Classifier(model=model,
                                      nb_classes=1000,
                                      loss_object=get(loss, kwargs.get('loss')),
                                      input_shape=get(image_size, kwargs.get('image_size')),
                                      clip_values=get((0,1), kwargs.get('clip_values')))
    return classifier

  if name == 'resnet50':
    class ResNet50Preprocessor(Preprocessor):
      def __call__(self, x, y=None):
        return resnet50.preprocess_input(x.copy()), y

      def estimate_gradient(self, x, gradient):
          return gradient[..., ::-1] 

    model = resnet50.ResNet50(weights='imagenet')
    classifier = KerasClassifier(model=model, 
                                 preprocessing=ResNet50Preprocessor(),
                                 clip_values=get((0,1), kwargs.get('clip_values')))

def get_predictions(model, X, **kwargs):
  predictions = model.predict(X)
  if kwargs.get('top_k'):
    k = kwargs.get('top_k')
    top_k = [np.argsort(prediction)[-k:][::-1] for prediction in predictions]
    return top_k
  
  return np.argmax(predictions, axis=1)

def get_accuracy(y, y_pred):
  accuracy = (np.sum(y_pred == y))/len(y)
  return accuracy

def jpeg_compress(X, quality_factor=50):
  X_compressed=[]
  
  for image in X:
    # Creating buffer space for saving image
    image_compressed = BytesIO()
    
    # Checking if the images passed are
    if np.max(image)<=1.0:
      image = to_image(image)

    # Compressing the image
    im1 = Image.fromarray(image.astype(np.uint8))
    im1.save(image_compressed, "JPEG", quality=quality_factor)
    
    X_compressed.append(np.asarray(Image.open(image_compressed)))
  
  return np.asarray(X_compressed)

def show_images(X=None, y=None, X_adv=None, y_adv=None, n=5):
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