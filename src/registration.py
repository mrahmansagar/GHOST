import SimpleITK as sitk
import numpy as np
from skimage import io
from tqdm import tqdm
from pathlib import Path
from typing import Optional


def preprocess_for_registration(fixed_image: np.ndarray, moving_image: np.ndarray) -> tuple[sitk.Image, sitk.Image]:
    """
    Preprocesses the input images for registration by averaging across channels (if needed) 
    and normalizing intensities.
    
    Args:
        fixed_image: nd array representing a fixed image that may have multiple channels (e.g., RGB).
        moving_image: nd array representing a moving image that may have multiple channels (e.g., RGB).
        The arrays are expected to be in the format (C, H, W) where C is the number of channels.
        If C=1, no averaging is performed.
        If C>1, channels are averaged to produce a single-channel image.
        The image values are expected to be in the range [0, 255] or [0, 1] and will be normalized to [0, 1].
        
    Returns:        tuple[sitk.Image, sitk.Image]: A tuple of preprocessed images ready for registration.
    """
    # Convert to numpy array for processing
    if np.ndim(fixed_image) == 3:
        fixed_img = np.average(fixed_image, axis=2)
    else:
        fixed_img = fixed_image

    if np.ndim(moving_image) == 3:
        # moving_img = np.average(moving_image, axis=2)
        moving_img = moving_image[:, :, 0]
    else:
        moving_img = moving_image


    fixed_img = fixed_img/np.max(fixed_img)*50.0
    moving_img = moving_img/np.max(moving_img)*50.0

    # Convert images to SimpleITK format if not already
    fixed_img = sitk.GetImageFromArray(fixed_img) if isinstance(fixed_img, np.ndarray) else fixed_img
    moving_img = sitk.GetImageFromArray(moving_img) if isinstance(moving_img, np.ndarray) else moving_img
    
    return fixed_img, moving_img



def compute_bspline_transform(fixed_image: sitk.Image, moving_image: sitk.Image, 
                            fixed_mask: Optional[sitk.Image] = None, moving_mask: Optional[sitk.Image] = None,
                            number_of_levels: int = 3, max_iter: int = 100) -> sitk.Transform:
    """
    Computes the B-Spline elastic transformation to align a moving image to a fixed image.
    
    Args:
        fixed_img: The reference SimpleITK image.
        moving_img: The SimpleITK image to be deformed.
        fixed_mask: Optional mask for the fixed image.
        moving_mask: Optional mask for the moving image.
        number_of_levels: Downsampling levels for multi-resolution registration.
        max_iter: Maximum iterations for the LBFGSB optimizer.
        
    Returns:
        sitk.Transform: The computed mathematical transformation.
    """
    
    # Initialize registration
    registration = sitk.ImageRegistrationMethod()

    # Metric: Mutual Information
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)

    # Optimizer: LBFGSB (Good for B-Spline)
    registration.SetOptimizerAsLBFGSB(numberOfIterations=max_iter)

    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # Setup B-spline Transform (elastic registration)
    grid_spacing = [15, 15]  # Control point spacing

    transform = sitk.BSplineTransformInitializer(fixed_image, grid_spacing)  # Adjust grid size as needed
    
    registration.SetInitialTransform(transform, inPlace=False)
  
    # Use a multi-level approach (each level corresponds to a different resolution)
    registration.SetShrinkFactorsPerLevel([4] * number_of_levels)  # Shrink factor for each level (downsampling factor)
    registration.SetSmoothingSigmasPerLevel([2] * number_of_levels)  # Smoothing sigma for each level

    # Use masks if provided
    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration.SetMetricMovingMask(moving_mask)

    # Progress bar setup
    pbar = tqdm(total=max_iter, desc=f"B-Spline Registration (Levels: {number_of_levels})", position=0)
    
    def update_progress():
        pbar.n = registration.GetOptimizerIteration()
        pbar.set_postfix(metric=registration.GetMetricValue())
        pbar.update(1)

    registration.AddCommand(sitk.sitkIterationEvent, update_progress)
    
    final_transform = registration.Execute(fixed_image, moving_image)
    pbar.close()
    
    return final_transform

def apply_transform(fixed_image: sitk.Image, moving_image: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """
    Applies a computed transform to a moving image and formats the output.
    
    Returns:
        np.ndarray: The aligned image as an 8-bit RGBA numpy array.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    
    resampled_image = resampler.Execute(moving_image)
    img_array = sitk.GetArrayFromImage(resampled_image)
    
    # Handle channel ordering (C, H, W) -> (H, W, C)
    if len(img_array.shape) == 3 and img_array.shape[0] == 3:  
        img_array = np.transpose(img_array, (1, 2, 0))
        
    # Normalize to 255
    img_array = (img_array / np.max(img_array)) * 255.0
    
    # Add alpha channel and mask background
    img_array = np.concatenate((img_array, np.ones((img_array.shape[0], img_array.shape[1], 1)) * 255), axis=-1)
    mask = np.all(img_array[:, :, :3] == img_array[0, 0, :3], axis=-1)
    img_array[mask, 3] = 0
    
    return img_array.astype('uint8')
    