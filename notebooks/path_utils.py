'''This script contains the implementation of the utilities that may be used for path changes.
Contributer: Bin Liu
Created Date: 2024-02-07
Last Updated: 2024-02-14
'''

import os
import sys

def add_parent_path_to_sys_path(path, verbose=False):
    '''This function adds the parent path of the input path to the sys path.
    
    Parameters:
    -----------
    path: str
        The input path.
    verbose: bool
        Whether to print the added path.
    '''
    # Get the parent path
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    # Add the parent path to the sys path
    sys.path.insert(0, parent_path)
    if verbose:
        print(f"Added {parent_path} to the sys path.")
    else:
        print('Path added to the sys path.')