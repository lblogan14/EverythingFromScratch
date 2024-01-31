'''This script contains the implementation of the image utilities that may be used in the repository.
Contributer: Bin Liu
Created Date: 2024-01-31
Last Updated: 2024-01-31
'''

from PIL import Image
import os
import glob
import matplotlib.pyplot as plt


def convert_RGB_to_Grayscale(img_rgb_folder, img_gray_folder, verbose=False):
    '''This function converts RGB images in the img_rgb_folder to grayscale images and save them in the img_gray_folder.
    
    Parameters:
    -----------
    img_rgb_folder: str
        The folder path of the RGB images.
    img_gray_folder: str
        The folder path to save the grayscale images.
    verbose: bool
        Whether to print the progress.
    '''

    # Create the folder if it doesn't exist
    if not os.path.exists(img_gray_folder):
        os.makedirs(img_gray_folder)

    # traverse all the images in the folder
    for subdir, dirs, files in os.walk(img_rgb_folder):
        for file in glob.glob(os.path.join(subdir, '*.png')):
            # Open the image
            img_rgb = Image.open(file)
            # Convert to grayscale
            img_gray = img_rgb.convert('L')
            # Save the grayscale image
            subdir_gray = subdir.replace(img_rgb_folder, img_gray_folder)
            if not os.path.exists(subdir_gray):
                # Create the folder if it doesn't exist
                os.makedirs(subdir_gray)
            img_gray.save(subdir_gray + '/' + os.path.basename(file))

            if verbose:
                print(f"Converted {file} to grayscale and saved in {img_gray_folder}.")
                fig, axes = plt.subplots(1, 2, figsize=(4,3))
                axes[0].imshow(img_rgb)
                axes[0].set_title('RGB')
                axes[0].axis('off')
                axes[1].imshow(img_gray, cmap='gray')
                axes[1].set_title('Grayscale')
                axes[1].axis('off')
                plt.show()

    # Print the completion message
    print(f"Converted all the images in {img_rgb_folder} to grayscale and saved in {img_gray_folder}.")