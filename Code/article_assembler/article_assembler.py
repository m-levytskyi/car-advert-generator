import os
import sys
import json
import subprocess
import re
from diffusers import StableDiffusionPipeline
import torch
import compel

import pypandoc


class ArticleAssembler:
    def __init__(self, template_file, output_dir="output", img_dir="imgs", tmp_dir="tmp"):
        """
        Initializes the ArticleAssembler.

        :param template_file: Path to the HTML/Markdown template file.
        :param output_dir: Directory to save output files.
        :param device: Device to run Stable Diffusion ("cpu" or "cuda").
        """
        self.template_file = template_file
        self.output_dir = output_dir
        self.img_dir = img_dir
        self.tmp_dir = tmp_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create required directories
        for directory in [output_dir, tmp_dir, img_dir]:
            os.makedirs(directory, exist_ok=True)
        print(f"Directories checked/created: {output_dir}, {tmp_dir}, {img_dir} ")

        self.pipeline = None

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
        

    @staticmethod
    def install_python_requirements():
        """
        Installs all Python libraries listed in requirements.txt.
        """
        try:
            print("Installing required Python libraries...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("All Python libraries installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while installing requirements: {e}")
            sys.exit(1)


    def load_json_data(self, json_file):
        """
        Loads data from a JSON file.

        :param json_file: Path to the JSON file.
        :return: List of dictionaries containing data.
        """
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            print(f"Data loaded successfully from {json_file}.")
            return data
        except FileNotFoundError:
            print(f"Error: {json_file} not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {json_file}.")
        return None

    def populate_template(self, paragraphs, captions, figure_paths, car_brand, car_model):
        """
        Populates the template with data.

        :param paragraphs: List of paragraphs.
        :param captions: List of captions.
        :param figure_paths: List of figure image paths.
        :param car_brand: Car brand name.
        :param car_model: Car model name.
        :return: Path to the populated HTML file.
        """
        with open(self.template_file, 'r') as file:
            template = file.read()

        # Fill placeholders
        content = {"car_brand": car_brand, "car_model": car_model}
        for i, paragraph in enumerate(paragraphs, start=1):
            content[f"paragraph_{i}"] = paragraph
        for i, caption in enumerate(captions, start=1):
            content[f"caption_{i}"] = caption
        for i, figure_path in enumerate(figure_paths, start=1):
            content[f"figure_{i}"] = f"../{figure_path}"

        for placeholder, replacement in content.items():
            template = re.sub(rf"\{{\{{\s*{re.escape(placeholder)}\s*\}}\}}", str(replacement), template)

        output_file = os.path.join(self.tmp_dir, "filled_article.html")
        with open(output_file, 'w') as file:
            file.write(template)
        print(f"Template populated successfully. Saved to {output_file}.")
        return output_file

    def setup_stable_diffusion(self):
        """
        Sets up the Stable Diffusion pipeline with device-specific optimizations.
        """
        print("Setting up Stable Diffusion...")
        
        if self.device == "cpu":
            print("Warning: Running on CPU. Image generation will be significantly slower.")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32
            )
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16
            )
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_attention_slicing()
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.pipeline = self.pipeline.to(self.device)
        print(f"Stable Diffusion setup complete on {self.device}.")

    def process_input(self, input_text, pipeline):
        compel_instance = compel.Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
        # Use compel to handle sequences longer than 77 tokens
        processed_text = compel_instance([input_text])
        return processed_text

    def generate_image(self, prompt, output_path):
        """
        Generates an image using Stable Diffusion.

        :param prompt: Text prompt for the image.
        :param output_path: Path to save the generated image.
        """
        if self.pipeline is None:
            raise ValueError("Stable Diffusion pipeline is not set up. Call `setup_stable_diffusion` first.")
        
        print(f"Generating image for prompt: {prompt}")
        prompt = self.process_input(prompt, self.pipeline)
        image = self.pipeline(prompt_embeds=prompt).images[0]
        image.save(output_path)
        print(f"Image saved to {output_path}")

    def convert_to_pdf(self, input_html, output_pdf):
        """
        Converts an HTML file to PDF using WeasyPrint.

        :param input_html: Path to the HTML file.
        :param output_pdf: Path to the output PDF file.
        """
        try:
            print("Converting HTML to PDF...")
            from weasyprint import HTML, CSS

            style_path = os.path.normpath("Code/article_assembler/styles.css")

            HTML(input_html).write_pdf(
                output_pdf, 
                stylesheets=[CSS(filename=style_path)]
                )
            print(f"PDF created successfully at {output_pdf}")
        
        except Exception as e:
            print(f"Error while converting PDF using weasyprint: {e}")
            try:
                print(f"Switching to xelatex...")
                # Get absolute paths
                abs_input = os.path.abspath(input_html)
                abs_output = os.path.abspath(output_pdf)
                resource_path = os.path.dirname(abs_input)
                
                # Configure logging
                import logging
                logging.getLogger('pypandoc').setLevel(logging.ERROR)
                
                # Convert with proper resource path
                pypandoc.convert_file(
                    abs_input,
                    to="pdf",
                    outputfile=abs_output,
                    extra_args=[
                        "--pdf-engine=xelatex",
                        f"--resource-path={resource_path}",
                        "--verbose"
                    ]
                )
                print(f"PDF created successfully at {output_pdf}")

            except Exception as e:
                print(f"Error during PDF creation: {e}")
                raise
        
        
        
