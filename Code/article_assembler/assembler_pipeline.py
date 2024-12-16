from .article_assembler import ArticleAssembler


class AssemblerPipeline:
    def __init__(self, json_path, images_path, output_pdf_path="Code/article_assembler/output/article.pdf", template_path="Code/article_assembler/article_template.html"):
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
        print("Loading car data...")
        car_data = self.assembler.load_json_data(self.json_path)

        if car_data:
            paragraphs = car_data["paragraphs"]
            prompts = car_data["prompts"]
            captions = car_data["captions"]
            car_brand = car_data["brand"]
            car_type = car_data["car_type"]

            print(f"Car Brand: {car_brand}, Car Type: {car_type}\n")

            # Setup Stable Diffusion
            print("Setting up Stable Diffusion...")
            self.assembler.setup_stable_diffusion()

            # Generate Images
            print("Generating images...")
            figure_paths = []
            for i, prompt in enumerate(prompts, start=1):
                fig_path = f"{self.images_path}/figure_{i}.png"
                print(f"Generating image {i} with prompt: '{prompt}'")
                self.assembler.generate_image(prompt, fig_path)
                figure_paths.append(fig_path)
            print("Image generation complete.\n")

            # Populate the template
            print("Populating the template...")
            html_file = self.assembler.populate_template(paragraphs, captions, figure_paths, car_brand, car_type)
            print("Template populated successfully.\n")

            # Convert HTML to PDF
            print("Converting to PDF...")
            self.assembler.convert_to_pdf(html_file, self.output_pdf_path)
            print(f"Assembler Pipeline complete. PDF saved at {self.output_pdf_path}")

        else:
            print("Error: Failed to load car data.")
