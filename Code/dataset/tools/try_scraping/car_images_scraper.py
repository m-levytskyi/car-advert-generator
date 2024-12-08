import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from PIL import Image
from io import BytesIO

# List of car brands to search
car_brands = ["HONDA", "LEXUS", "FIAT", "PEUGEOT", "VOLVO", "OPEL", "MAZDA", "ALFAROMEO", "CITROEN", "SKODA"]

# Directory to save the images
output_dir = "/Users/johannesdecker/CAR_SCRAPING"
os.makedirs(output_dir, exist_ok=True)

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
service = Service("/Users/johannesdecker/chromedriver-mac-arm64/chromedriver")  # Replace with your chromedriver path

driver = webdriver.Chrome(service=service, options=chrome_options)

# Helper function to download and save an image
def download_image(url, save_path):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
    except Exception as e:
        print(f"Failed to download image: {e}")

# Function to scrape images for a specific car brand
def scrape_images(brand, limit=10):
    search_query = f"{brand} car exterior"
    driver.get("https://www.google.com/imghp")

    # Handle pop-ups or overlays (e.g., cookie consent)
    try:
        consent_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I agree') or contains(text(), 'Accept')]"))
        )
        consent_button.click()
    except Exception as e:
        print("No consent button found or other pop-up skipped:", e)

    # Search for the query
    try:
        search_box = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "q"))
        )
        driver.execute_script("arguments[0].focus();", search_box)  # Focus on the search box
        driver.execute_script(f"arguments[0].value = '{search_query}';", search_box)
        search_box.send_keys(Keys.RETURN)
    except Exception as e:
        print(f"Search box interaction failed, injecting query directly: {e}")
        try:
            driver.execute_script(
                f"document.querySelector('[name=q]').value = '{search_query}';"
            )
            driver.execute_script(
                "document.querySelector('[name=q]').dispatchEvent(new Event('input'));"
            )
            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys(Keys.RETURN)
        except Exception as js_error:
            print(f"JavaScript fallback also failed: {js_error}")
            return

    # Wait for images to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "rg_i")))

    # Get image elements
    images = driver.find_elements(By.CLASS_NAME, "rg_i")

    count = 0
    for img in images:
        if count >= limit:
            break

        try:
            img.click()
            time.sleep(1)  # Wait for the larger image to load

            # Get the larger image URL
            large_image = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "n3VNCb"))
            )
            image_url = large_image.get_attribute("src")

            if image_url and image_url.startswith("http"):
                save_path = os.path.join(output_dir, f"{brand}_{count + 1}.jpg")
                download_image(image_url, save_path)
                print(f"Downloaded: {save_path}")
                count += 1
        except Exception as e:
            print(f"Error scraping image for {brand}: {e}")

# Main scraping logic
try:
    for brand in car_brands:
        brand_dir = os.path.join(output_dir, brand)
        os.makedirs(brand_dir, exist_ok=True)
        scrape_images(brand, limit=10)
finally:
    driver.quit()

print("Scraping completed!")
