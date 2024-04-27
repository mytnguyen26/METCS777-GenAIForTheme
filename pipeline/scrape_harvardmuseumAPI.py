import requests
import os
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API key
API_KEY = os.getenv("API_KEY")

# Base URL of the Harvard Art Museums API
BASE_URL = 'https://api.harvardartmuseums.org'

# Endpoint for searching objects
SEARCH_ENDPOINT = '/object'

# Parameters for your search
params = {
    'apikey': API_KEY,
    'classification': 'Paintings',  # Filter by classification (Paintings)
    'keyword': 'Chinese ink',       # Filter by keyword (Chinese ink)
    'size': 5000                     # Number of results to retrieve
}

def fetch_data(endpoint, params):
    try:
        response = requests.get(BASE_URL + endpoint, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print('Error:', e)

def download_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the image file
        with open(filename, 'wb') as f:
            f.write(response.content)
        print('Downloaded:', filename)
    except requests.exceptions.RequestException as e:
        print('Error downloading image:', e)

def main():
    # Fetch data from the API
    data = fetch_data(SEARCH_ENDPOINT, params)
    # Specify the folder to save images
    save_folder = 'images'

    if data: 
        # Specify the CSV filename
        csv_filename = 'harvard_metadata.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data['records'][0].keys())
            writer.writeheader()
            # Process the data
            for artwork in data['records']:
                # Check if the artwork has a primaryimageurl and it's not None
                if artwork.get('primaryimageurl') is not None:
                    writer.writerow(artwork)
                    # Print the title of the artwork
                    print('Title:', artwork['title'])
                    # Construct the download link for the image
                    download_link = artwork['primaryimageurl'] + '?height=500'  # Example: Resize image to height of 500px
                    print('Download link:', download_link)

                    # Extract filename from the URL
                    filename = os.path.join(save_folder, str(artwork['id']) + '.jpg')

                    # Download the image file
                    download_image(download_link, filename)
                    print()  # Add a newline for readability
        

if __name__ == '__main__':
    main()
