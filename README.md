# work-effi-gen
### Install

Install dependencies for running with Python 3 and GPU enabled Tensorflow

```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
$HOME/.local/bin/pip3 install virtualenv --user

$HOME/.local/bin/virtualenv ecg_env
source ecg_env/bin/activate # add to .bashrc.user

pip install -r path_to/requirements.txt

## Add below to .bashrc.user
# for cuda 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64

# for cuda nvcc
export PATH=$PATH:/usr/local/cuda-8.0/bin:
```

### Run

Run with
```
gpu=0
env CUDA_VISIBLE_DEVICES=$gpu python train.py
```

### Tensorboard

To view results run:
```
tensorboard --port 8888 --logdir <directory_of_saved_models>
```

### Jupyter Notebook

First install `jupyter` with
```
pip install jupyter
```

Then to launch the notebook

```
cd notebooks
env CUDA_VISIBLE_DEVICES=<gpu> jupyter notebook --port <port> --ip 0.0.0.0
```
replace `<gpu>` and `<port>` with desired values.

