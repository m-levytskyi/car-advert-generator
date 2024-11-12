import subprocess
import sys
import os

def install_pandoc_and_latex():
    try:
        subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Pandoc is already installed.")
    except FileNotFoundError:
        print("Installing Pandoc...")
        if sys.platform.startswith("linux"):
            subprocess.run(["sudo", "apt-get", "install", "-y", "pandoc"])
        elif sys.platform == "darwin":
            subprocess.run(["brew", "install", "pandoc"])
        elif sys.platform == "win32":
            print("Please install Pandoc manually from https://pandoc.org/installing.html for Windows.")
    latex_installed = os.system("xelatex -version") == 0
    if not latex_installed:
        print("Installing LaTeX...")
        if sys.platform.startswith("linux"):
            subprocess.run(["sudo", "apt-get", "install", "-y", "texlive-xetex", "texlive-fonts-recommended", "texlive-fonts-extra"])
        elif sys.platform == "darwin":
            subprocess.run(["brew", "install", "mactex-no-gui"])
        elif sys.platform == "win32":
            print("Please install LaTeX manually from https://miktex.org/download for Windows.")
