# Image-Classification-Model-2
Build Deep Learning model for the Image Classification task (Part2): Structure, Monitoring for training.

Dependencies 
+ OS: Mac/Linux/Window
+ Python3 (version >= 3.7) [Download](https://www.python.org/downloads/)

## 1. Setup 

Conda environment 

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv)$ pip install -r requirements.txt
```

## 2. Run script  

Train CNN model 

```bash 
(venv)$ python3 train.py --config config/config.yml
```

Evaluate the trained model

```bash 
(venv)$ python3 test.py --config config/config.yml
```

Predict pretrained model on an image 
```bash
(venv)$ python3 predict.py --config config/config.yml --image data/test/airplane/30.png
```


## 3. Monitoring 

To view Tensorboard, start 
```bash
(venv)$ tensorboard --logdir=./logs/tensorboard
# Check http://localhost:6006
```

To view MLflow Dashboard, start 
```bash
(venv)$ mlflow ui
# Check http://localhost:5000
```