# Create virtual env
'''
ssh 192.93.8.192
mkdir venv
cd venv
virtualenv -p python3.5 diatoms3.5
. diatoms3.5/bin/activate
python -m pip install ipython
python -m pip install jupyter lab
python -m pip install tensorflow
python -m pip install tensorflow-gpu
python -m pip install imutils
python -m pip install opencv-python
python -m pip install Cython contextlib2 pillow lxml matplotlib jupyter
'''

# Installing object detection API
'''
python -m pip install object-detection
'''
THEN
'''
git clone https://github.com/tensorflow/models.git
cd research
python setup.py install
'''
eg. 
export PYTHONPATH=$PYTHONPATH:/home/GTL/pfauregi/venv/obj_dect/models/research:/home/GTL/pfauregi/venv/obj_dect/models/research/slim
