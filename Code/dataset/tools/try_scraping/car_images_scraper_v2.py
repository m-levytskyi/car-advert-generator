import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# List of car brands to search for
car_brands = [
    'honda', 'lexus', 'fiat', 'peugeot', 'volvo',
    'opel', 'mazda', 'alfaromeo', 'citroen', 'skoda'
]

# Directory to save downloaded images
output_dir = "/Users/johannesdecker/CAR_SCRAPING"
os.makedirs(output_dir, exist_ok=True)

# User-Agent header to mimic a real browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Keywords to identify interior images
interior_keywords = ['interior', 'dashboard', 'cabin', 'seat', 'steering']

def is_exterior_image(image_url):
    """
    Checks if the image URL likely corresponds to an exterior image.
    """
    return not any(keyword in image_url.lower() for keyword in interior_keywords)

def download_image(url, brand):
    """
    Downloads and saves the image from the given URL.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        # Save image with a unique name
        image_filename = os.path.join(output_dir, f"{brand}_{os.path.basename(url)}")
        image.save(image_filename)
        print(f"Downloaded {image_filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def scrape_images_for_brand(brand):
    """
    Scrapes exterior images for the specified car brand.
    """
    search_query = f"{brand} car exterior"
    search_url = f"https://www.google.com/search?tbm=isch&q={search_query}"
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tags = soup.find_all('img')
        for img in image_tags:
            img_url = img.get('src')
            if img_url and is_exterior_image(img_url):
                download_image(img_url, brand)
    except Exception as e:
        print(f"Failed to scrape images for {brand}: {e}")

if __name__ == "__main__":
    for brand in car_brands:
        scrape_images_for_brand(brand)
