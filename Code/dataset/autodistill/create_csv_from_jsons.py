import json
import pandas as pd

# Load JSON data from the uploaded files
body_styles_path = '/Users/johannesdecker/ADL_DATASETS/DS2/body_styles_confidence_image_labels.json'
brands_path = '/Users/johannesdecker/ADL_DATASETS/DS2/brands_confidence_image_labels.json'

with open(body_styles_path, 'r') as f:
    body_styles_data = json.load(f)

with open(brands_path, 'r') as f:
    brands_data = json.load(f)

# Create a DataFrame combining data from both JSON files
images = list(body_styles_data.keys())
brands = [brands_data[image] for image in images]
body_styles = [body_styles_data[image] for image in images]


# data = {'image': images, 'brand': brands, 'body_style': body_styles}
confidence_body_styles = [body_styles_data[image] for image in images]
confidence_brands = [body_styles_data[image] for image in images]
data = {'image': images, 'brand': brands, 'body_style': body_styles,
        'confidence_brands': confidence_brands, 'confidence_body_styles': confidence_body_styles}


df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_path = '/Users/johannesdecker/ADL_DATASETS/DS2/matched_image_data.csv'
df.to_csv(output_path, index=False)

