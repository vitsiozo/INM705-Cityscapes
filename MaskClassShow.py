
import os
from PIL import Image
import numpy as np

# Define the directory path
dir_path = '/Users/vitsiozo/Code/Image/dataset/gtCoarse/train/dusseldorf/'

# Get a list of all files in the directory
files = os.listdir(dir_path)

# Initialize an empty set to store all unique values
all_unique_values = set()

# Iterate over the files
for file in files:
    # Check if the file ends with 'labelIds.png'
    if file.endswith('labelIds.png'):
        # Construct the full file path
        file_path = os.path.join(dir_path, file)
        
        # Open the image file
        with Image.open(file_path) as img:
            # Convert the image to a numpy array
            img_array = np.array(img)
            
            # Get the unique values in the array
            unique_values = np.unique(img_array)

            # Add the unique values to the set
            all_unique_values.update(unique_values)
            
            # Print the unique values
            print(f'Unique values for {file}: {unique_values}')

# Print the total number of unique values across all images
print(f'Values across all images: {all_unique_values}, Total values: {len(all_unique_values)}')