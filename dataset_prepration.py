import os
import shutil
import random
import pandas as pd
from PIL import Image
import numpy as np
import tifffile
import shutil




def train_val_split(original_folder, train_folder, val_folder, val_ratio=0.2):
    
    """
    Splits original folder of images into train and validation folder.
    val and train folders are specified in config.yaml
    """
        
    # Create train and val folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # List all files in the original folder
    all_files = os.listdir(original_folder)
    
    # Shuffle the list of files (optional)
    random.shuffle(all_files)
    
    # Calculate split indices
    split_index = int(0.8 * len(all_files))  # 80% for training
    
    # Move files to train folder
    for filename in all_files[:split_index]:
        src = os.path.join(original_folder, filename)
        dst = os.path.join(train_folder, filename)
        shutil.move(src, dst)
    
    # Move files to val folder
    for filename in all_files[split_index:]:
        src = os.path.join(original_folder, filename)
        dst = os.path.join(val_folder, filename)
        shutil.move(src, dst)




def convert_df_to_yolo_format(csv_file, labels_path):


    """
    Potato dataset consists of xml files, therefore we need to convert them to yolov8 acceptable format
    """
    df = pd.read_csv(csv_file)
    # Group DataFrame by image filename
    grouped = df.groupby('filename')

    # Iterate over each image filename group
    for filename, group in grouped:
        filename = filename.replace("images/", "")

        # Create an empty list to store YOLO annotations for the current image
        annotations = []

        # Iterate over rows in the current image filename group
        for index, row in group.iterrows():
            # Extract bounding box coordinates and class label from the row
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            class_name = row['class']
            if class_name == "stressed":
              class_name = '1'
            if class_name == "healthy":
              class_name = '0'

            # Convert bounding box coordinates to YOLO format
            x_center = (xmin + xmax) / (2 * 750)
            y_center = (ymin + ymax) / (2 * 750)
            width = (xmax - xmin) / 750
            height = (ymax - ymin) / 750

            # Append YOLO annotation to the list
            annotations.append(f'{class_name} {x_center} {y_center} {width} {height}')

        # Save YOLO annotations to a text file for the current image filename
        with open(os.path.join(labels_path, os.path.splitext(filename)[0] + '.txt'), 'w') as f:
            f.write('\n'.join(annotations))









def combine_channels(combination_type, green_folder, red_folder, near_infrared_folder, red_edge_folder, output_folder):

    """
    Potato dataset multi spectral channels are in a form of single channels, we need to merge them to build a 4 channel image.
    """

    combination_mapping = {
        "RGREN": ["red", "green", "red_edge", "near_infrared"],
        "RGN": ["red", "green", "near_infrared"],
        "RGE": ["red", "green", "red_edge"],
        # Add more combination types here as needed
    }

    if combination_type not in combination_mapping:
        print("Invalid combination type")
        return

    channels = combination_mapping[combination_type]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(green_folder):
        if filename.endswith(".jpg"):  # assuming images are in jpg format

                # Open images from each band
                green_image = Image.open(os.path.join(green_folder, filename))
                red_image = Image.open(os.path.join(red_folder, filename))

                red_array = np.array(red_image)
                green_array = np.array(green_image)

                # Merge channels based on combination type
                if combination_type == 'RGREN':
                    near_infrared_image = Image.open(os.path.join(near_infrared_folder, filename))
                    red_edge_image = Image.open(os.path.join(red_edge_folder, filename))

                    near_infrared_array = np.array(near_infrared_image)
                    red_edge_array = np.array(red_edge_image)

                    merged_array = np.stack((red_array, green_array, red_edge_array, near_infrared_array), axis=-1)

                elif combination_type == 'RGN':
                    near_infrared_image = Image.open(os.path.join(near_infrared_folder, filename))

                    near_infrared_array = np.array(near_infrared_image)

                    merged_array = np.stack((red_array, green_array, near_infrared_array), axis=-1)

                elif combination_type == 'RGE':
                    red_edge_image = Image.open(os.path.join(red_edge_folder, filename))

                    red_edge_array = np.array(red_edge_image)

                    merged_array = np.stack((red_array, green_array, red_edge_array), axis=-1)

                # Construct the output path for the merged image
                output_filename = os.path.splitext(filename)[0] + ".tif"
                output_path = os.path.join(output_folder, output_filename)

                # Save the merged image directly to TIFF format
                tifffile.imwrite(output_path, merged_array)


def move_matching_labels(base_folder, origin_folder, destination_folder):
    """
    Move files from the origin folder that match names with files in the base folder to the destination folder.
    
    Args:
        base_folder (str): Path to the base folder.
        origin_folder (str): Path to the folder containing files to be checked for matches.
        destination_folder (str): Path to the folder where matching files will be moved.
    """
    # Get list of files in the base folder
    base_files = os.listdir(base_folder)
    
    # Get list of files in the origin folder
    origin_files = os.listdir(origin_folder)
    
    # Iterate over files in the origin folder
    for file_name in origin_files:
        # Check if the file name matches any of the files in the base folder
        label_filename, _ = os.path.splitext(file_name)
        # Check if the label file name matches any of the image file names in the val folder
        if label_filename in [os.path.splitext(file)[0] for file in base_files]:
            # Move the file from the origin folder to the destination folder
            src = os.path.join(origin_folder, file_name)
            dst = os.path.join(destination_folder, file_name)
            shutil.move(src, dst)
            print(f"Moved {file_name} to {destination_folder}")



def rename_labels_add_underline(labels_folder):
    # Get a list of all files in the labels folder
    label_files = os.listdir(labels_folder)
    
    # Iterate over each label file
    for filename in label_files:
        # Split the filename into its base and extension parts
        base, ext = os.path.splitext(filename)
        
        # Remove the "image" prefix and the ".txt" extension to get the numeric part
        numeric_part = base.replace("Image", "")
        
        # Construct the new filename with the desired format
        new_filename = f"Image_{numeric_part}{ext}"
        
        # Construct the full paths for the old and new filenames
        old_path = os.path.join(labels_folder, filename)
        new_path = os.path.join(labels_folder, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)


def rename_labels_to_uppercase(folder_path):
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        # Construct the current and new filenames
        current_path = os.path.join(folder_path, filename)
        new_filename = filename.capitalize()  # Convert first letter to uppercase
        
        # Check if the new filename is different from the current filename
        if new_filename != filename:
            new_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(current_path, new_path)
            print(f"Renamed {filename} to {new_filename}")