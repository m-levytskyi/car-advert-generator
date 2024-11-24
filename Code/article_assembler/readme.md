# Article Assembler


## Installation of Dependencies

Should be done automatically when running `assembler_pipeline.py`. Otherwise read the instruction below.

### 1. Python Dependencies

Are listed in `requirements.txt`

### 2. System-Level Dependencies

**pandoc**, **wkhtmltopdf**, and **LaTeX**

The installation is implemented in `article_assembler.py`. 

```python
    @staticmethod
    def install_dependencies():
        def run_command(command):
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {' '.join(command)}\n{e.stderr.decode()}")
                sys.exit(1)
        
        # Check if Pandoc is installed
        try:
            subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Pandoc is already installed.")
        except FileNotFoundError:
            print("Installing Pandoc...")
            if sys.platform.startswith("linux"):
                if os.path.exists("/etc/arch-release"):
                    run_command(["sudo", "pacman", "-Syu", "--needed", "pandoc"])
                else:
                    run_command(["sudo", "apt-get", "install", "-y", "pandoc"])
            elif sys.platform == "darwin":
                run_command(["brew", "install", "pandoc"])
            elif sys.platform == "win32":
                print("Please install Pandoc manually from https://pandoc.org/installing.html for Windows.")
        
        # Check if LaTeX is installed
        latex_installed = os.system("xelatex -version") == 0
        if not latex_installed:
            print("Installing LaTeX...")
            if sys.platform.startswith("linux"):
                if os.path.exists("/etc/arch-release"):
                    run_command(["sudo", "pacman", "-Syu", "--needed", "texlive-core", "texlive-fontsextra"])
                else:
                    run_command(["sudo", "apt-get", "install", "-y", "texlive-xetex", "texlive-fonts-recommended", "texlive-fonts-extra"])
            elif sys.platform == "darwin":
                run_command(["brew", "install", "mactex-no-gui"])
            elif sys.platform == "win32":
                print("Please install LaTeX manually from https://miktex.org/download for Windows.")
        
        # Check if wkhtmltopdf is installed
        try:
            subprocess.run(["wkhtmltopdf", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("wkhtmltopdf is already installed.")
        except FileNotFoundError:
            print("Installing wkhtmltopdf...")
            if sys.platform.startswith("linux"):
                if os.path.exists("/etc/arch-release"):
                    run_command(["sudo", "yay", "-S", "wkhtmltopdf"])
                else:
                    run_command(["sudo", "apt-get", "install", "-y", "wkhtmltopdf"])
            elif sys.platform == "darwin":
                run_command(["brew", "install", "wkhtmltopdf"])
            elif sys.platform == "win32":
                print("Please install wkhtmltopdf manually from https://wkhtmltopdf.org/downloads.html for Windows.")
```

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