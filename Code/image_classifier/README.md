- install Python 3.12
- navigate to /Code/image_classifier
- create venv with python 3.12
- enter venv
- pip install -r requirements_cuda.txt 
    (alternatively install compatible Torch version according to your graphics card 
    -> replace --index-url https://download.pytorch.org/whl/cu124 with --index-url https://download.pytorch.org/whl/cu121 or --index-url https://download.pytorch.org/whl/cu118
    OR use CPU version - no indexurl , but not recommended)
- pip install -r requirements.txt
- (in /Code/image_classifier): python train.py
