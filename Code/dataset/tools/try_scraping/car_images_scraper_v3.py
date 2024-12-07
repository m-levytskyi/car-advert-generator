from GoogleImageScraper import GoogleImageScraper
import os

# Define parameters
search_keys = ['car']
number_of_images = 10
headless = False
min_resolution = (800, 600)
max_resolution = (1920, 1080)
output_dir = "/Users/johannesdecker/CAR_SCRAPING"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize scraper
scraper = GoogleImageScraper(
    webdriver_path='/Users/johannesdecker/chromedriver-mac-arm64/chromedriver',
    image_path=output_dir,
    search_key=search_keys[0],
    number_of_images=number_of_images,
    headless=headless,
    min_resolution=min_resolution,
    max_resolution=max_resolution
)

# Start scraping
scraper.find_image_urls()
scraper.save_images()
