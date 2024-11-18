# Article Assembler


## Installation

Should be done automatically when running `main.py`. Otherwise read the instruction below.

### 1. Python Dependencies

Are listed in `requirements.txt`

### 2. System-Level Dependencies

**pandoc**, **wkhtmltopdf**, and **LaTeX**

The installation implemented in `dependencies.py`. 
In case it doesnt work:

Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y pandoc wkhtmltopdf texlive-xetex texlive-fonts-recommended texlive-fonts-extra
```


Arch Linux

```bash
sudo pacman -Syu --needed pandoc texlive-core texlive-fontsextra
yay -S wkhtmltopdf  # Requires AUR helper like `yay`
```


macOS

1. Install Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. Install dependencies:

```bash
brew install pandoc wkhtmltopdf
brew install --cask mactex-no-gui
```

Windows

Install Pandoc:
Download and install from [Pandoc Official Website](https://pandoc.org/installing.html).

Install LaTeX:
Download and install MiKTeX from [MiKTeX Official Website](https://miktex.org/download).

Install wkhtmltopdf:
Download and install from [wkhtmltopdf Official Website](https://wkhtmltopdf.org/downloads.html).