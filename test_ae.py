import sys, getopt
from ae import AE
from ae import Attacks

########################### CONSTANTS ###########################
epsilon = 0.1
eps_step = epsilon/10
attack_type = Attacks.FGSM
prefix = '/content/drive/MyDrive/AE_Resources/Imagenette'
image_size = (300, 300, 3)
per_class = 100
max_iter = 10
#################################################################

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

image_size_str = str(image_size[0]) + '_' + str(image_size[1]) + '_' + str(image_size[2])
epsilon_str = str(epsilon).split('.')[0] + "_" + str(epsilon).split('.')[1]

params = {
  'epsilon': epsilon,
  'epsilon_str': epsilon_str,
  'eps_step': eps_step,
  'attack_type': attack_type,
  'prefix': prefix,
  'image_size': image_size,
  'image_size_str': image_size_str,
  'per_class': per_class,
  'max_iter': max_iter,
}

ae = AE(params)
