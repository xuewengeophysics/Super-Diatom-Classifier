ssh 192.93.8.192
mkdir venv
cd venv

# Create virtual env tf2.1
'''
virtualenv -p python3.5 p3.5tf2.1 
. diatoms3.5/bin/activate  
python -m pip install ipython  
python -m pip install jupyterlab  
python -m pip install tensorflow==2.1.0  
python -m pip install tensorflow-gpu==2.1.0  
python -m pip install imutils  
python -m pip install opencv-python  
python -m pip install Cython contextlib2 pillow lxml matplotlib  
'''

# Create virtual env tf1.15
'''
virtualenv -p python3.5 p3.5tf1.15  
. p3.5tf1.15/bin/activate  
python -m pip install ipython  
python -m pip install jupyterlab  
python -m pip install tensorflow==1.15.2  
python -m pip install tensorflow-gpu==1.15.2  
python -m pip install imutils  
python -m pip install opencv-python  
python -m pip install Cython contextlib2 pillow lxml matplotlib  
'''

# Installing object detection API
'''
git clone https://github.com/tensorflow/models.git
cd research
'''
eg. 
export PYTHONPATH=$PYTHONPATH:/home/GTL/pfauregi/venv/obj_dect/models/research:/home/GTL/pfauregi/venv/obj_dect/models/research/slim
