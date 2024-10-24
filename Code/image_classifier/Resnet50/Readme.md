## How to run Resnet50 Transfer Learning

### Step 0: Install the required libraries
- Install the required libraries using the command `pip install -r requirements.txt`
- Path to the requirements file: `Code/image_classifier/Resnet50/requirements.txt`

### Step 1: Get the data
- Download the data from [here](https://drive.google.com/file/d/1T7B2vLKL7g6PI4ebMxvN8ALzJr213fUB/view?usp=sharing)
- Extract the data in the directory `Code/dataset/Data/`
- e.g. Path to final_2.csv should be `Code/dataset/Data/DS1_vorläufig_Car_Models_3778/final_2.csv`

### Step 2: Prepare the data
- Run `Code/image_classifier/Resnet50/dataloader.py` once to get the data in the required format
- This will create a `correct.csv` file in the directory (`Code/dataset/Data/DS1_vorläufig_Car_Models_3778/correct.csv`)
- This file will be used for getting the correct paths to the images
- Note: Run everything from the root directory of the project (`adl-gruppe-1`)

### Step 3: Train the model
- Run `Code/image_classifier/Resnet50/main.py`
- Optional: Set Flag `load_everything_in_memory` to `True` in `main.py` to load the data in memory (Note: this is untested)