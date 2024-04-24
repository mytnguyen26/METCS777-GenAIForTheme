import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time


def fetch_page(id):
    url = f'https://digicol.dpm.org.cn/?page={id}&category=17'
    chromium_path = "/Users/caozhen/Downloads/chrome-mac/Chromium"
    options = webdriver.ChromeOptions()
    options.binary_location = chromium_path
    options.headless = True

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(1)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    images = soup.find_all('div', class_='pic')
    paintings_info = []
    for image in images:
        name = image.get('aria-label')
        img_box = image.find("div", class_="img_box2") or image.find("div", class_="img_box3")
        cultural_id = img_box.get('id')
        img = image.find("img", class_="img001") or image.find("img", class_="img002")
        image_url = img.get('src')

        paintings_info.append({
            'name': name,
            'id': cultural_id,
            'image_url': image_url
        })

    return paintings_info


def fetch_all(start_page=1):
    csv_file = 'palace_museum.csv'
    df = pd.read_csv(csv_file)

    # fetch all pages
    page = start_page
    while True:
        print(f'Fetching page {page}...')
        paintings_info = fetch_page(page)
        if len(paintings_info) == 0: break

        for info in paintings_info:
            id = info.get('id', None)
            title = info.get('name', None)
            image_url = info.get('image_url', None)
            # Downloading the image
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                # Save the image
                with open(f"palace_paintings/{id}.png", 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {id} successfully.")
            except requests.RequestException as e:
                print(f"Failed to download {id}. Reason: {str(e)}")

            row = {'file_name': id, 'caption': title}
            df = df._append(row, ignore_index=True)

        df.to_csv(csv_file, index=False)
        # next page
        page += 1


if __name__ == '__main__':
    fetch_all(start_page=208)
