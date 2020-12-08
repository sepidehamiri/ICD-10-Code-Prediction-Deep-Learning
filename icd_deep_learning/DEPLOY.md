
hamester deployment guide
==================================


### Prerequisites


```bash
sudo apt install build-essential libncursesw5-dev libreadline6-dev \
 libssl-dev libgdbm-dev libc6-dev libsqlite3-dev tk-dev sqlite3
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
#### Adding a system user and its directories.

```bash
sudo useradd dev
su - dev
```

#### install Virtual env

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
```

#### install requirements 
```bash
pip3.6 install -r requirements.txt
```
#### Geting Data from Author: Sepideh Shamsizadeh

#### Define the data folder paths
spesify the data folder path in `hamster.yml` file as:
```bash
word2vector_filename: word-vectors/word2vector.txt
dataset_filename: dataset/dataset.csv
stop_words_filename: stop-words/stopwords.txt
deep_learning_directory: deep_model/
```
#### Set text for predicting it's labels:
In `main.py`, define a text and two thresholds for confidence and value of attention.

Run `main.py`

You can visulize the result in visualization.html


