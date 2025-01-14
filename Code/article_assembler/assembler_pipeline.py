from .article_assembler import ArticleAssembler

import time

class AssemblerPipeline:
    def __init__(self, json_path, images_path, output_pdf_path="Code/article_assembler/output/article.pdf", template_path="Code/article_assembler/nolatex_article_template.html"):
        """
        Initializes the pipeline with paths and configuration.

        :param template_path: Path to the HTML/Markdown template file.
        :param json_path: Path to the JSON file containing car data.
        :param images_path: Directory to store generated images.
        :param output_pdf_path: Path to save the final PDF.
        """
        self.template_path = template_path
        self.json_path = json_path
        self.images_path = images_path
        self.output_pdf_path = output_pdf_path

        self.assembler = ArticleAssembler(template_file=self.template_path, img_dir=self.images_path)

    def setup(self, py_reqs=True, os_reqs=True):
        """
        Sets up the environment by installing dependencies and Python requirements.
        """
        print("Setting up the environment...")
        if os_reqs:
            self.assembler.install_dependencies()

        if py_reqs:    
            self.assembler.install_python_requirements()
        print("Setup complete.\n")

    def run_pipeline(self):
        """
        Executes the entire pipeline: loading data, generating images, populating the template, and creating the PDF.
        """
        total_start = time.time()

        # Step 1: Load Data
        load_start = time.time()
        print("Loading car data...")
        car_data = self.assembler.load_json_data(self.json_path)
        load_time = time.time() - load_start

        if car_data:
            paragraphs = car_data["paragraphs"]
            prompts = car_data["prompts"]
            captions = car_data["captions"]
            car_brand = car_data["brand"]
            car_type = car_data["car_type"]

            print(f"Car Brand: {car_brand}, Car Type: {car_type}\n")

            # Setup Stable Diffusion
            setup_start = time.time()
            print("Setting up Stable Diffusion...")
            self.assembler.setup_stable_diffusion()
            setup_time = time.time() - setup_start

            # Generate Images
            image_gen_start = time.time()
            print("Generating images...")
            figure_paths = []
            for i, prompt in enumerate(prompts, start=1):
                fig_path = f"{self.images_path}/figure_{i}.png"
                print(f"Generating image {i} with prompt: '{prompt}'")
                self.assembler.generate_image(prompt, fig_path)
                figure_paths.append(fig_path)

            image_gen_time = time.time() - image_gen_start
            num_images = len(prompts)
            avg_image_time = image_gen_time / num_images
            print("Image generation completed.\n")

            # Populate the template
            template_start = time.time()
            print("Populating the template...")
            html_file = self.assembler.populate_template(paragraphs, captions, figure_paths, car_brand, car_type)
            template_time = time.time() - template_start
            print("Template populated successfully.\n")

            # Convert HTML to PDF
            pdf_start = time.time()
            print("Converting to PDF...")
            self.assembler.convert_to_pdf(html_file, self.output_pdf_path)
            print("\n\n==================================")
            print(f"Assembler Pipeline complete. PDF saved in root directory as {self.output_pdf_path}")
            print("==================================\n\n")
            pdf_time = time.time() - pdf_start

            total_time = time.time() - total_start

            # Print timing summary
            print("=== Pipeline Timing Summary ===")
            print(f"JSON Data Loading: {load_time:.2f} seconds")
            print(f"Stable Diffusion Setup: {setup_time:.2f} seconds")
            print(f"Image Generation: {image_gen_time:.2f} seconds")
            print(f"Average time per image: {avg_image_time:.2f} seconds")
            print(f"Template Population: {template_time:.2f} seconds")
            print(f"PDF Conversion: {pdf_time:.2f} seconds")
            print(f"Total Pipeline Time: {total_time:.2f} seconds")
            print("===========================\n")

        else:
            print("Error: Failed to load car data.")
