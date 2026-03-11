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
    # Ensure fixed image is 3D (H, W, C)
    if fixed_img_array.ndim == 2:
        # Convert grayscale to RGB by stacking
        fixed_img_array = np.stack([fixed_img_array] * 3, axis=-1)
    elif fixed_img_array.shape[2] > 3:
        # If it has an alpha channel, strip it for the overlay
        fixed_img_array = fixed_img_array[:, :, :3]

    # Initialize the checkerboard with 3 channels
    h, w = registered_img_array.shape[:2]
    checker = np.zeros((h, w, 3), dtype='uint8')
    
    # Normalize fixed image to 0-255
    fixed_norm = (fixed_img_array / (np.max(fixed_img_array) + 1e-5) * 255.0).astype('uint8')
    
    # Take only RGB from registered (ignore Alpha)
    reg_rgb = registered_img_array[:, :, :3]

    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            r_end = min(r + block_size, h)
            c_end = min(c + block_size, w)
            
            if ((r // block_size) + (c // block_size)) % 2 == 0:
                checker[r:r_end, c:c_end, :] = reg_rgb[r:r_end, c:c_end, :]
            else:
                checker[r:r_end, c:c_end, :] = fixed_norm[r:r_end, c:c_end, :]
                
    return checker


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
