import re

def populate_template(template_file, paragraphs, captions, figure_paths, car_brand, car_model):
    # Read the template file
    with open(template_file, 'r') as file:
        template = file.read()

    # Basic placeholders for car brand and model
    content = {
        "car_brand": car_brand,
        "car_model": car_model
    }

    # Dynamically add paragraph placeholders
    for i, paragraph in enumerate(paragraphs, start=1):
        content[f"paragraph_{i}"] = paragraph

    # Dynamically add caption placeholders
    for i, caption in enumerate(captions, start=1):
        content[f"caption_{i}"] = caption

    # Dynamically add figure placeholders
    for i, figure_path in enumerate(figure_paths, start=1):
        content[f"fig{i}"] = figure_path

    # Replace each placeholder in the template using regular expressions
    for placeholder, replacement in content.items():
        template = re.sub(rf"\{{\{{\s*{placeholder}\s*\}}\}}", replacement, template)

    # Save the populated content to a new markdown file
    with open("filled_article.md", "w") as file:
        file.write(template)
