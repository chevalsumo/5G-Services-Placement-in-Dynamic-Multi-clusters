apt update -y 
apt install xz-utils -y
#Installation de submariner sur le control plane du cluster "key"
#curl -Ls https://get.submariner.io | bash

curl -Ls https://get.submariner.io > ~/submariner.sh

bash ~/submariner.sh
export PATH=$PATH:~/.local/bin
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc

