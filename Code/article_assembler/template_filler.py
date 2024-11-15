import re

def populate_template(template_file, paragraphs, captions, figure_paths, car_brand, car_model):
    # Read the template file
    try:
        with open(template_file, 'r') as file:
            template = file.read()
    except FileNotFoundError:
        print(f"Error: Template file '{template_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading template file: {e}")
        return

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

    print(figure_paths)
    # Dynamically add figure placeholders
    for i, figure_path in enumerate(figure_paths, start=1):
        content[f"figure_{i}"] = figure_path

    # Replace each placeholder in the template using regular expressions
    for placeholder, replacement in content.items():
        # Use re.escape() to handle any special characters in placeholders
        template = re.sub(rf"\{{\{{\s*{re.escape(placeholder)}\s*\}}\}}", str(replacement), template)

    # Save the populated content to a new HTML file
    try:
        with open("filled_article.html", "w") as file:
            file.write(template)
        print("Template populated successfully and saved as 'filled_article.html'")
    except Exception as e:
        print(f"Error writing populated HTML: {e}")