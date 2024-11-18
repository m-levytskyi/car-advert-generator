import subprocess
import sys
import os

def install_system_dependencies():
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

def install_python_requirements():
    """
    Installs all Python libraries listed in requirements.txt.
    """
    try:
        print("Installing required Python libraries...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "article_assembler/requirements.txt"])
        print("All Python libraries installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing requirements: {e}")
        sys.exit(1)


