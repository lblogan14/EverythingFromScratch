'''This script contains the implementation of the image utilities that may be used in the repository.
Contributer: Bin Liu
Created Date: 2024-01-31
Last Updated: 2024-01-31
'''

from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display_html
import numpy as np


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


def display_numpy_arrays_as_images():
    '''This function enables the display of numpy arrays as images in Jupyter Notebook.
    '''
    
    def array2png(arr):
        '''This function converts a numpy array to a PNG image.
        
        Parameters:
        -----------
        arr: numpy array
            The input numpy array.
        
        Returns:
        --------
        PIL.Image
            The PNG image.
        '''
        # Check if the input is a 2D or 3D array
        if 2 <= len(arr.shape) <= 3:
            return Image.fromarray(np.array(np.clip(arr, 0, 1) * 255,
                                            dtype=np.uint8))._repr_png_()
        else:
            # If the input is not a 2D or 3D array, return a blank image
            return Image.fromarray(np.zeros([1, 1],
                                            dtype=np.uint8))._repr_png_()
        
    def array2text(obj, p, cycle):
        if len(obj.shape) < 2:
            print(repr(obj))
        
        if 2 <= len(obj.shape) <= 3:
            pass
        else:
            print(f'<array of shape {obj.shape}>')

    # Register the display function
    get_ipython().display_formatter.formatters['image/png'].for_type(np.ndarray, array2png)
    get_ipython().display_formatter.formatters['text/plain'].for_type(np.ndarray, array2text)



# Display the answer in a hoverable way
_style_inline = """<style>
.einops-answer {
    color: transparent;
    padding: 5px 15px;
    background-color: #def;
}
.einops-answer:hover { color: blue; } 
</style>
"""
def show_tensors_as_html(x):
    display_html(
        _style_inline
        + "<h4>Answer is: <span class='einops-answer'>{x}</span> (hover to see)</h4>".format(x=tuple(x)),
        raw=True)