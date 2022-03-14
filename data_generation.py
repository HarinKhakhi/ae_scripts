import sys, getopt
from ae import AE
from ae import Attacks

from tensorflow.keras.layers import Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import inception_v3

from art.estimators.classification import TensorFlowV2Classifier
########################### CONSTANTS ###########################
epsilon = 0.1
eps_step = epsilon/10
attack_type = Attacks.FGSM.name
prefix = '/content/drive/MyDrive/AE_Resources/Imagenette/Datasets'
image_size = (300, 300, 3)
per_class = 100
max_iter = 10
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
  opts, args = getopt.getopt(sys.argv[1:],"he:a:s:n:p:",["epsilon=","attack=","save_image=","per_class=", "path="])
except getopt.GetoptError:
  print('script.py --epsilon <epsilon> --attack <attack> --path <path_to_directory>')
  sys.exit(2)
for opt, arg in opts:
  if opt in ('-e', '--epsilon'):
    epsilon = float(arg)
  if opt in ('-a', '--attack'):
    attack_type = arg
  if opt in ('-s', '--saveImage'):
    save_image = bool(arg)
  if opt in ('-n', '--perClass'):
    per_lass = int(arg)
  if opt in ('-p', '--path'):
    prefix = arg
  if opt in ('-h'):
    print('script.py --epsilon <epsilon> --attack <attack> --path <path_to_directory>')
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
if save_image:
  ae.create_adv_dataset(X, ae.params['class_list'], saveImage=False)
  X_adv = ae.get_adv_dataset(fromImage=False)
else:
  X_adv = ae.create_adv_dataset(X, ae.params['class_list'], saveImage=False)

y_adv = ae.get_predictions(X_adv)

adv_accuracy = ae.get_accuracy(X_adv, org_class)
print('Adversarial Accuracy :', adv_accuracy)