import numpy as np
import SimpleITK as sitk

def create_checkerboard_overlay(
    fixed_img_array: np.ndarray, 
    registered_img_array: np.array, 
    block_size: int = 50
) -> np.ndarray:
    """
    Creates a checkerboard pattern alternating between the fixed and registered images.
    Args:
        fixed_img_array: The original fixed image as a numpy array.
        registered_img_array: The registered image as a numpy array.
        block_size: The size of the checkerboard blocks in pixels.
    Returns:
        A numpy array representing the checkerboard overlay.    
    """
    checker = np.zeros((registered_img_array.shape[0], registered_img_array.shape[1], 3))
    fixed_normalized = (fixed_img_array / np.max(fixed_img_array)) * 255.0
    
    for r in range(0, checker.shape[0], block_size):
        for c in range(0, checker.shape[1], block_size):
            if ((r // block_size) + (c // block_size)) % 2 == 0:
                checker[r:r+block_size, c:c+block_size, :] = registered_img_array[r:r+block_size, c:c+block_size, :3]
            else:
                checker[r:r+block_size, c:c+block_size, :] = fixed_normalized[r:r+block_size, c:c+block_size, :3]
                
    return checker.astype('uint8')


def create_deformed_grid(
    fixed_img: sitk.Image, 
    moving_img: sitk.Image, 
    transform: sitk.Transform, 
    line_spacing: int = 20
) -> np.ndarray:
    """
    Creates a grid of lines, deforms them using the provided transform, 
    and returns the resulting image.
    """

    for r in range(0, moving_img.GetSize()[1], line_spacing):
        for x in range(moving_img.GetSize()[0]):
            moving_img.SetPixel(x, r, (0, 0, 0))  # Set to black
    for c in range(0, moving_img.GetSize()[0], line_spacing):
        for y in range(moving_img.GetSize()[1]):
            moving_img.SetPixel(c, y, (0, 0, 0))    

    # Resample the grid using the computed transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    deformed_grid_sitk = resampler.Execute(moving_img)
    
    # Convert to numpy and normalize for saving
    grid_array = sitk.GetArrayFromImage(deformed_grid_sitk)
    if len(grid_array.shape) == 3 and grid_array.shape[0] == 3:  
        grid_array = np.transpose(grid_array, (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
    grid_array = grid_array/np.max(grid_array)*255.0
    grid_array=grid_array.astype('uint8')
    
    return grid_array
