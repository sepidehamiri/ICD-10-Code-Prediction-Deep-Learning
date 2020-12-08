# ICD-10 Code Prediction Deep Learning


hamster 
========

Automatic Medical Coding with Deep Neural Networks.


Setting up development Environment on Linux:
-------

### Prerequisites


```bash
sudo apt install build-essential libncursesw5-dev libreadline6-dev libssl-dev \
  libgdbm-dev libc6-dev libbz2-dev libsqlite3-dev tk-dev sqlite3
```

### Python3.6

```bash
sudo apt update && sudo apt build-dep python3.5
cd /tmp
wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz
tar -xvf Python-3.6.*.tgz
cd Python-3.6.*
./configure
make -j 8
sudo make altinstall
```

### Preparing the install location

#### Install Virtual environment

```bash
sudo pip3.6 install virtualenvwrapper
``` 

#### Setup virtual environment
```bash
echo "export VIRTUALENVWRAPPER_PYTHON=`which python3.6`" >> ~/.bashrc
echo "alias v.activate=\"source $(which virtualenvwrapper.sh)\"" >> ~/.bashrc
source ~/.bashrc
v.activate
mkvirtualenv --python=$(which python3.6) --no-site-packages hamster
```
#### Activating virtual environment

```bash
workon hamster
```

#### Cloning

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:Carrene/hamster.git
cd hamster
pip install -e .
```
#### Geting Data from R&D Team

#### Command-line Interface:
```bash
hamster -h
```
```bash
usage: hamster [-h] [-c CONFIDENCE] [-t THRESHOLD] manifest text

Automatic Medical Coding with Deep Neural Networks.

positional arguments:
  manifest              Address of manifest.yml function
  text                  input medical text to predict ICD-9 codes

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIDENCE, --confidence CONFIDENCE
                        The confidence threshold (default = 0.5)
  -t THRESHOLD, --threshold THRESHOLD
                        The attention threshold (default = 0.2)

```

#### Project usage:

To use the project import `Predictor` from `hamster.code_prediction`: 

```bash
from hamster.code_prediction import Predictor
```
make an object of `Predictor` by pass the manifest file path to it: 

```bash
predictor = Predictor(manifest_path)
```
and call the `predict_code` with the following parameters:
```bash
codes = predictor.predict_code(medical_text, confidence, threshold) 
```



