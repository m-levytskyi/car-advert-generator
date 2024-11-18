from json_loader import load_car_info
from template_filler import populate_template
from image_generation import generate_image
from markdown_to_pdf import convert_md_to_pdf
from dependencies import install_pandoc_and_latex

############################################
## CHANGE ACCORDINGLY ##
template_path = "article_template.html"
json_path = "../article_agent/output.json"
###########################################

# Install dependencies
print("Checking and installing dependencies...")
install_pandoc_and_latex()
print("Dependencies check complete.\n")

car_data = load_car_info(json_path)

#########
car_brand = 'BMW' #car_data["car_brand"]
car_model = 'X5' #car_data["car_model"]
#########

paragraphs = car_data["paragraphs"]
prompts = car_data["prompts"]
captions = car_data["captions"]

figure_paths = []

for i, prompt in enumerate(prompts, start=1):
    image_path = f"imgs/figure_{i}.png"
    print(f"Generating image {i} with prompt: '{prompt}'")
    generate_image(prompt, output_path=image_path)
    figure_paths.append(image_path)
print("Image generation complete.\n")

# Populate the template
print("Populating the article template with provided content...")
populate_template(template_path, paragraphs, captions, figure_paths, car_brand, car_model)

# Convert to PDF
print("Converting the populated article to PDF...")
convert_md_to_pdf("tmp/filled_article.html", "article.pdf")
print("PDF conversion complete. Check 'article.pdf' for the output.")