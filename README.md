# README

## Prerequisites

-   Python 3.11 - 3.12
-   CUDA-capable GPU with at least 4GB VRAM (recommended)
-   Webcam (recommended)

## Installation

1.  **Clone the repository:**

``` bash
git clone https://gitlab.lrz.de/simon-hampp/adl-gruppe-1.git
cd adl-gruppe-1
```

2.  **Create a virtual environment:**

``` bash
python -m venv venv
source venv/bin/activate
```

3.  **Install dependencies:**

-   If using a CUDA-Compatible GPU (optional, but running a Stable Diffusion on CPU may take up to 1h):

``` bash
pip install -r requirements_cuda.txt
```

-   General:

``` bash
pip install -r requirements.txt
```

4.  **Make sure Pandoc, GTK3 and MikTeX (xelatex) are installed and added to the PATH.**

It should happen automatically if you are using Linux.

Otherwise you can download it from the following link:

**Pandoc**: https://pandoc.org/installing.html
**GTK3**: 
Windows: github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
MacOS: https://www.gtk.org/docs/installations/macos/
**MiKTeX**: https://miktex.org/download

## Usage

1.  **Navigate to project root:**

``` bash
cd path/to/adl-gruppe-1
```

2.  **Run the pipeline:**

``` bash
python Code/pipeline.py [--mode <camera|images>] [--images_path /path/to/images]
```

Arguments:

-   `--mode:` Specifies the mode to run the pipeline. It can be either `camera` or `images`.

    -   `camera`: Uses the webcam to capture images. This is the default mode and can be omitted if you want to use the webcam.

    -   `images`: Uses pre-existing images from a specified directory.

-   `--images_path`: Specifies the path to the directory containing images to be used instead of capturing new ones with the webcam. This argument is optional and defaults to **Code/webcam/demo_images**, which contains images of a BMW X5. Use this argument if you want to specify a different directory.

Examples:

-   To run the pipeline using the webcam (default mode):

``` bash
python Code/pipeline.py
```

-   To run the pipeline using images from the default demo directory:

``` bash
python Code/pipeline.py --mode images
```

-   To run the pipeline using images from a specific directory:

``` bash
python Code/pipeline.py --mode images --images_path /path/to/images
```

3.  **Follow the interactive prompts (if in `camera` mode):**

-   Webcam window will open
-   Press SPACE to capture car images
-   Press ENTER when finished capturing
-   Press ESC to abort

**Wait for processing.**

## Output locations:

-   **Final Article**: Code/article.pdf
-   Captured images: Code/webcam/webcam_images/
-   Processed images: Code/webcam/processed_images/
