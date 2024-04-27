import requests
import os
import pandas as pd
from configs import DATA_FOLDER, OUTPUT_FOLDER

# Set the directory path and file name
metObj_dir = DATA_FOLDER
metObj_file = os.path.join(metObj_dir, 'MetObjects' + '.txt')

try:
    # Attempt to read the text file into a DataFrame
    all_paintings_df = pd.read_csv(metObj_file, low_memory=False, sep='\t')  # Assuming tab-separated values
    print('Opened data file:', metObj_file)

except Exception as e:
    # Print an error message if opening the file fails
    print(e)
    print('Failed to open data file')

# Filter paintings from China
filtered_paintings = all_paintings_df[
    all_paintings_df['Classification'].isin(['Painting', 'Paintings']) &
    (all_paintings_df['Culture'] == 'China')
]

filtered_paintings['Title'] = filtered_paintings['Title'].str.split('|').str[1]

try:
    # Specify the directory path and file name for the output text file
    output_dir = OUTPUT_FOLDER
    txt_file = os.path.join(output_dir, 'metropolitan_metadata.txt')

    # Export the column to a text file
    filtered_paintings['Object Number'].to_csv(txt_file, index=False, header=False)  # Assuming you don't want to include index or header

    print(f'Exported Object Number to {txt_file}')

except Exception as e:
    print(e)
    print('Failed to export column to text file')

try:
    # Specify the directory path and file name for the output CSV file
    csv_file = os.path.join(output_dir, 'painting.csv')

    # Write the DataFrame to a CSV file
    filtered_paintings.to_csv(csv_file, index=False)

    print(f'Exported DataFrame to {csv_file}')

except Exception as e:
    print(e)
    print('Failed to export DataFrame to CSV file')


def download_images_from_api(object_ids, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Base URL for the API
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"

    # Iterate over each object ID and download its image
    for object_id in object_ids:
        # Construct the URL for the specific object
        url = base_url + str(object_id)

        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Check if the object has a primary image
            if 'primaryImage' in data:
                img_url = data['primaryImage']
                img_name = f"image_{object_id}.jpg"
                img_path = os.path.join(output_dir, img_name)

                try:
                    # Download the image
                    img_data = requests.get(img_url).content
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_data)
                    print(f"Downloaded image for object {object_id}")
                except Exception as e:
                    print(f"Failed to download image for object {object_id}: {e}")
            else:
                print(f"No primary image found for object {object_id}")
        else:
            print(f"Failed to fetch data for object {object_id}")


object_ids = filtered_paintings['Object ID'].tolist()
output_dir = "met_images"

download_images_from_api(object_ids, output_dir)
