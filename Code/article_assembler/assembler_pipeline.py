from article_assembler import ArticleAssembler

############################################
## CHANGE ACCORDINGLY ##
template_path = "article_template.html"
json_path = "../article_agent/json/output_toyota_coupe.json" #article agent output
images_path = "tmp/imgs" #generated images will be stored here
###########################################

# Initialize the assembler

assembler = ArticleAssembler(template_file=template_path, 
                            img_dir=images_path)


# Install dependencies
assembler.install_dependencies()

assembler.install_python_requirements()

# Load data
car_data = assembler.load_json_data(json_path)

if car_data:

    paragraphs = car_data["paragraphs"]
    prompts = car_data["prompts"]
    captions = car_data["captions"]

    car_brand = car_data["brand"]
    car_type = car_data["car_type"]

    # Setup Stable Diffusion
    assembler.setup_stable_diffusion()

    # TODO:
    # stable diffusion takes only the first 77 tokens of the prompt
    # possible fix - compel
    # https://github.com/damian0815/compel/tree/main/doc

    # Generate Images
    figure_paths = []
    for i, prompt in enumerate(prompts, start=1):
        fig_path = f"{images_path}/figure_{i}.png"
        print(f"Generating image {i} with prompt: '{prompt}'")
        assembler.generate_image(prompt, fig_path)
        figure_paths.append(fig_path)
    print("Image generation complete.\n")


    print('\n\n\n', figure_paths, '\n\n\n')
    # Populate template
    html_file = assembler.populate_template(paragraphs, captions, figure_paths, car_brand, car_type)

    # Convert HTML to PDF
    assembler.convert_to_pdf(html_file, "output/article.pdf")
