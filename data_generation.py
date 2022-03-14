import sys, getopt
from ae import AE
from ae import Attacks

from tensorflow.keras.layers import Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import inception_v3

from art.estimators.classification import TensorFlowV2Classifier
########################### INIT CONSTANTS ###########################
epsilon = 0.1
eps_step = epsilon/10
attack_type = Attacks.FGSM.name
prefix = '/content/drive/MyDrive/AE_Resources/Imagenette/Datasets'
image_size = (300, 300, 3)
per_class = 100
max_iter = 100
class_start_index = 0
class_end_index = 10
#################################################################
model = inception_v3.InceptionV3( input_tensor=Input(shape=image_size),
                                  include_top=True, 
                                  weights='imagenet', 
                                  classifier_activation='softmax')
loss = CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model,
                                    nb_classes=1000,
                                    loss_object=loss,
                                    input_shape=image_size,
                                    clip_values=(-1,1))
########################### ARGUMENTS ###########################
try:
  opts, args = getopt.getopt(sys.argv[1:],"ha:e:t:n:p:i:j:",["attack=","epsilon=","max_iter=","per_class=", "path=", "class_start=", "class_end="])
except getopt.GetoptError:
  print('script.py \
      --attack <attack> \
      --epsilon <epsilon> \
      --max_iter <int> \
      --per_class <int> \
      --path <path_to_directory> \
      --class_start <int> \
      --class_end <int>')
  sys.exit(2)
for opt, arg in opts:
  if opt in ('-a', '--attack'):
    attack_type = arg
  if opt in ('-e', '--epsilon'):
    epsilon = float(arg)
  if opt in ('-t', '--max_iter'):
    max_iter = int(arg)  
  if opt in ('-n', '--per_class'):
    per_lass = int(arg)
  if opt in ('-p', '--path'):
    prefix = arg
  if opt in ('-i', '--class_start'):
    class_start_index = int(arg)
  if opt in ('-j', '--class_end'):
    class_end_index = int(arg)
  if opt in ('-h'):
    print('script.py \
      --attack <attack> \
      --epsilon <epsilon> \
      --max_iter <int> \
      --per_class <int> \
      --path <path_to_directory> \
      --class_start <int> \
      --class_end <int>')
    sys.exit()
#################################################################

params = {
  'epsilon': epsilon,
  'eps_step': eps_step,
  'attack_type': attack_type,
  'prefix': prefix,
  'image_size': image_size,
  'per_class': per_class,
  'max_iter': max_iter,
}

ae = AE(params, classifier)

# Loading Data
X, org_class = ae.get_org_dataset(fromImage=False)

org_accuracy = ae.get_accuracy(X, org_class)
print('Original Accuracy :', org_accuracy)

# Generating Adversarial Data
ae.create_adv_dataset(X, ae.params['class_list'][class_start_index: class_end_index])