import requests
from PIL import Image
from io import BytesIO
import json
import csv
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API key
api_key = os.getenv("API_KEY")

# Function to transform the sample data
def transform_data(sample_data):
    transformed_data = {}

    # Copying unchanged fields
    transformed_data['id'] = sample_data['id']
    transformed_data['title'] = sample_data['title']
    transformed_data['timestamp'] = sample_data['timestamp']
    transformed_data['lastTimeUpdated'] = sample_data['lastTimeUpdated']
    transformed_data['version'] = sample_data['version']

    # Transforming content section
    content = sample_data['content']
    transformed_data['freetext'] = content['freetext']
    transformed_data['indexedStructured'] = content['indexedStructured']

    # Transforming descriptiveNonRepeating section
    descriptive_non_repeating = content['descriptiveNonRepeating']
    transformed_data['guid'] = descriptive_non_repeating['guid']
    transformed_data['record_ID'] = descriptive_non_repeating['record_ID']
    transformed_data['unit_code'] = descriptive_non_repeating['unit_code']
    transformed_data['title_sort'] = descriptive_non_repeating['title_sort']
    transformed_data['data_source'] = descriptive_non_repeating['data_source']
    transformed_data['record_link'] = descriptive_non_repeating['record_link']
    transformed_data['metadata_usage'] = descriptive_non_repeating['metadata_usage']

    # Transforming online_media section
    online_media = descriptive_non_repeating['online_media']
    if 'media' in online_media and online_media['media']:
        resources = online_media['media'][0]['resources']
        for resource in resources:
            if resource['label'] == 'High-resolution JPEG':
                transformed_data['high_resolution_jpeg_url'] = resource['url']
                break

    return transformed_data

# Function to search content based on a query with maximum rows
def search_content_max_rows(query, start=0, rows=1000, sort='relevancy', type='edanmdm', row_group='objects'):
    url = 'https://api.si.edu/openaccess/api/v1.0/search'
    params = {
        'q': query,
        'start': start,
        'rows': rows,  # Adjusted to fetch maximum rows per request
        'sort': sort,
        'type': type,
        'row_group': row_group,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to download image from URL
def download_image_from_url(url, filename, save_folder):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the response content as an image using PIL
        image = Image.open(BytesIO(response.content))
        
        # Save the image to the specified folder
        filepath = os.path.join(save_folder, filename)
        image.save(filepath)
        print(f"Image downloaded and saved as '{filepath}'")
    else:
        print("Failed to download image")

def main():
    # Specify the folder to save images
    save_folder = 'images'

    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Perform the search - keywords = "Chinese Art" & "Painting"
    query = 'topic:"Chinese Art" AND objectType:"Painting"'
    category_content = search_content_max_rows(query, start="ld1-1643390182193-1643390184728-0")

    # Check if the request was successful
    if 'response' in category_content:
        data = category_content['response']['rows']
        transformed_data = [transform_data(data[i]) for i in range(len(data))]

        # Specify the CSV filename
        csv_filename = 'smithsonian_content.csv'

        # Open a CSV file for writing
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=transformed_data[0].keys())
            writer.writeheader()

            for item in transformed_data:
                # Write the transformed item to CSV
                writer.writerow(item)

                # Check if the item has an image URL
                if 'high_resolution_jpeg_url' in item:
                    # Extract image URL
                    image_url = item['high_resolution_jpeg_url']

                    # Extract filename from the URL
                    filename = item['id'] + '.jpg'

                    # Download the image using the download_image_from_url function
                    download_image_from_url(image_url, filename, save_folder)
    else:
        print("Error: Unable to fetch data.")

if __name__ == '__main__':
    main()
