#!/usr/bin/env bash

pip install adversarial-robustness-toolbox > /dev/null
rm -rf /content/ae_scripts
git clone https://github.com/HarinKhakhi/ae_scripts.git /content/ae_scripts/ > /dev/null
echo "Setup Complete"