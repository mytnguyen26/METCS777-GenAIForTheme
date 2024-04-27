import os
import pandas as pd
from configs import BASE_FOLDER

# Set the directory path and file name
base_dir = BASE_FOLDER

# Harvard 
harvObj_dir = os.path.join(base_dir, 'Harvard', 'chinese ink')

# Metropolitan 
metObj_dir = os.path.join(base_dir, 'Metropolitan')

# Smithsonian
smisoObj_dir = os.path.join(base_dir, 'SmithSonian')

try:   
    # Read Harvard images and metadata file into a DataFrame
    harv_images = [image_nm for image_nm in os.listdir(harvObj_dir)
                   if image_nm.endswith("jpg")]
    print('Opened Harvard image files:', len(harv_images))

    harv_metadata_file = [file for file in os.listdir(harvObj_dir) if file.endswith("csv")][0]
    harv_df = pd.read_csv(os.path.join(harvObj_dir, harv_metadata_file))
    print('Opened Harvard metadata file: ', harv_df.shape)

    # Read Metropolitan images and metadata file into a DataFrame
    met_images_dir = os.path.join(metObj_dir, 'met_images')
    met_images = [image_nm for image_nm in os.listdir(met_images_dir)
                  if image_nm.endswith("jpg")]
    print('Opened Metropolitan image files:', len(met_images))

    met_df = pd.read_csv(os.path.join(metObj_dir, 'painting.csv'))
    print('Opened Metropolitan metadata file: ', met_df.shape)

    # Read Smithsonian images and metadata file into a DataFrame
    smiso_images_dir = os.path.join(smisoObj_dir, 'images')
    smiso_images = [image_nm for image_nm in os.listdir(smiso_images_dir)
                    if image_nm.endswith("jpg")]
    print('Opened Smithsonian image files:', len(smiso_images))

    smiso_df = pd.read_csv(os.path.join(smisoObj_dir, 'smithsonian_content.csv'))
    print('Opened Smithsonian metadata file: ', smiso_df.shape)

except Exception as e:
    # Print an error message if opening the file fails
    print(e)
    print('Failed to open data file')

# Harvard Metadata Cleaning
harv_cols = ["id", "title", "period"]
harv_df["caption"] = harv_df[["title", "period"]].apply(lambda x: f"{x.title} completed in {x.period}", axis=1)
harv_df["file_name"] = harv_df.id.apply(lambda x: f"{x}.jpg")
harv_metadata_df = harv_df[["file_name", "caption"]]

# Metropolitan Metadata Cleaning
filtered_met_df = met_df[met_df['Classification'].isin(['Painting', 'Paintings']) & 
                         (met_df['Culture'] == 'China')]
filtered_met_df['file_name'] = 'image_' + filtered_met_df['Object ID'].astype(str) + '.jpg'
filtered_met_df = filtered_met_df[['file_name'] + [col for col in met_df.columns]]

cleaned_met_df = filtered_met_df.copy()
cleaned_met_df['Title'] = cleaned_met_df['Title'].str.split('|').str[0].fillna('')
cleaned_met_df['Period'] = cleaned_met_df['Period'].apply(lambda x: 'in the ' + x if not pd.isna(x) else '')
temp = cleaned_met_df['Tags'].apply(lambda x: x if not pd.isna(x) else '').str.split('|')
temp = temp.apply(lambda x: [tag for tag in x])
temp = temp.apply(lambda x: ' and '.join(x) if len(x) > 1 else (x[0] if x else ''))
cleaned_met_df['Tags'] = temp
cleaned_met_df['Object Name'] = cleaned_met_df['Object Name'].apply(lambda x: 'in a ' + x if not pd.isna(x) else '')
cleaned_met_df['caption'] = cleaned_met_df['Tags'].where(cleaned_met_df['Tags'] != '', cleaned_met_df['Title'])

# Smithsonian Metadata Cleaning
smiso_df['id'] = smiso_df['id'].apply(lambda x: x + '.jpg' if not pd.isna(x) else '')
smiso_df['title'] = smiso_df['title'].apply(lambda x: x if not pd.isna(x) else '')
smiso_metadata_df = pd.DataFrame({
    'file_name': smiso_df['id'],
    'caption': smiso_df['title']
})

# Combine all metadata
harv_final_metadata = harv_metadata_df[harv_metadata_df['file_name'].isin(harv_images)]
metro_final_metadata = cleaned_met_df[cleaned_met_df['file_name'].isin(met_images)]
smiso_final_metadata = smiso_metadata_df[smiso_metadata_df['file_name'].isin(smiso_images)]
combined_metadata_df = pd.concat([harv_final_metadata, metro_final_metadata, smiso_final_metadata], ignore_index=True)

# Export to CSV
new_metadata_file = os.path.join(base_dir, 'metadata.csv')
combined_metadata_df.to_csv(new_metadata_file, index=False)
print("DataFrame exported to metadata.csv")
