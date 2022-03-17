#!/bin/bash

# install requirements
echo "Installing requirements"
pip3 install -r requirements.txt --user
pip3 install git+https://github.com/gstaff/flask-ngrok

# download CETENfolha
echo "Downloading CETENFolha"
file="CETENFolha-1.0.cg"

cd files

if [ -f "$file" ]
then
	echo "$file already downloaded."
else
	wget https://www.linguateca.pt/cetenfolha/download/CETENFolha-1.0.cg.gz --no-check-certificate
    gzip -d CETENFolha-1.0.cg.gz
    echo "Download completed\n"
fi

## Download model and tokenizer
echo "Download model and tokenizer"
gdown https://drive.google.com/uc?id=1cixSE_KPDgJ0q7vGrjGC-t7aNGv37YTn
unzip trained-bertimbau-1503.zip
gdown https://drive.google.com/uc?id=1yA8cdTezDqe0rIc-oDw0wUzxk1YLSJqK
