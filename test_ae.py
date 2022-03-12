import sys, getopt
from ae import AE
from ae import Attacks

########################### CONSTANTS ###########################
epsilon = 0.1
eps_step = epsilon/10
attack_type = Attacks.FGSM.name
prefix = '/content/drive/MyDrive/AE_Resources/Imagenette'
image_size = (300, 300, 3)
per_class = 100
max_iter = 10
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
#################################################################
model = {}
loss = {}
classifier = {}
########################### ARGUMENTS ###########################
try:
  opts, args = getopt.getopt(sys.argv[1:],"he:a:p:",["epsilon=","attack=","path="])
except getopt.GetoptError:
  print('script.py --epsilon <epsilon> --attack <attack> --path <path_to_directory>')
  sys.exit(2)
for opt, arg in opts:
  if opt in ('-e', '--epsilon'):
    epsilon = float(arg)
  if opt in ('-a', '--attack'):
    attack_type = arg
  if opt in ('-p', '--path'):
    prefix = arg
  if opt in ('-h'):
    print('script.py --epsilon <epsilon> --attack <attack> --path <path_to_directory>')
#################################################################

params = {
  'epsilon': epsilon,
  'epsilon_str': epsilon_str,
  'eps_step': eps_step,
  'attack_type': attack_type,
  'prefix': prefix,
  'image_size': image_size,
  'per_class': per_class,
  'max_iter': max_iter,
}

ae = AE(params, classifier)