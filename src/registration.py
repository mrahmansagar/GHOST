import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage import io
import shutil
from pathlib import Path, PurePath


def register_2D_image(fixed_file, moving_file, fixed_mask=None, moving_mask=None, 
                      number_of_levels=3, max_iter=100, checkers=None, lines=None,
                      processed_dir=None, copy_originals=True):


    fixed_image = io.imread(fixed_file).astype('float')
    moving_image = io.imread(moving_file).astype('float')


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

    if fixed_mask == None:
        fixed_mask = np.ones((fixed_img.shape[0], fixed_img.shape[1]))
    else:
        fixed_mask = fixed_mask

    if moving_mask == None:
        moving_mask = np.ones((moving_img.shape[0],moving_img.shape[1]))
    else:
        moving_mask = moving_mask
    
    # Create a progress bar
    pbar = tqdm(total=100, desc="B-Spline Registration Progress Level ["+str(number_of_levels)+"]", position=0)
    
    def update_progress(registration_method):
        """Update tqdm progress bar based on iteration"""
        iteration = registration_method.GetOptimizerIteration()
        metric = registration_method.GetMetricValue()
    
        pbar.n = iteration  # Update iteration count
        pbar.set_postfix(metric=metric)  # Show metric value
        pbar.update(1)  # Update the bar
    
    """ Perform B-Spline based elastic registration with optional masks. """
    
    # Convert images to SimpleITK format if not already
    fixed_img = sitk.GetImageFromArray(fixed_img) if isinstance(fixed_img, np.ndarray) else fixed_img
    moving_img = sitk.GetImageFromArray(moving_img) if isinstance(moving_img, np.ndarray) else moving_img
    
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

    transform = sitk.BSplineTransformInitializer(fixed_img, grid_spacing)  # Adjust grid size as needed
    
    registration.SetInitialTransform(transform, inPlace=False)
  
    # Use a multi-level approach (each level corresponds to a different resolution)
    registration.SetShrinkFactorsPerLevel([4] * number_of_levels)  # Shrink factor for each level (downsampling factor)
    registration.SetSmoothingSigmasPerLevel([2] * number_of_levels)  # Smoothing sigma for each level

    # Use masks if provided
    try:
        registration.SetMetricFixedMask(fixed_mask)
    except:
        ...
    try:
        registration.SetMetricMovingMask(moving_mask)
    except:
        ...

    # Attach progress observer
    registration.AddCommand(sitk.sitkIterationEvent, lambda: update_progress(registration))
    
    final_transform = registration.Execute(fixed_img, moving_img)

    fixed_img = sitk.ReadImage(fixed_file)
    moving_img = sitk.ReadImage(moving_file)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    resampled_image = resampler.Execute(moving_img)


    RI=sitk.GetArrayFromImage(resampled_image)
    if len(RI.shape) == 3 and RI.shape[0] == 3:  
        RI = np.transpose(RI, (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
    RI = RI/np.max(RI)*255
    RI = np.concatenate((RI,np.ones((RI.shape[0],RI.shape[1],1))*255),axis=-1)
    mask = np.all(RI[:,:,:3]==RI[0,0,:3],axis=-1)
    RI[mask,3]=0
    RI=RI.astype('uint8')

    # Saving all the files 
    if processed_dir == None:
        save_dir = "./registration_output"
    else:
        save_dir = processed_dir

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Registered file
    registered_fname = f"{save_dir}/{PurePath(moving_file).stem}_registered.tif"
    print('Saving deformed images as ',registered_fname)
    io.imsave(registered_fname, RI)

    
    if copy_originals:
    # Fixed File
        shutil.copy2(fixed_file, save_dir)

        # Moving file
        shutil.copy2(moving_file, save_dir)  

    # Checker board
    if checkers is not None:
        checker = np.zeros((RI.shape[0],RI.shape[1],3))
        # F = io.imread(fixed_img_file_name).astype('float')
        F =  fixed_image
        F = F/np.max(F)*255

        
        block_size = checkers
        for r in range(0, RI.shape[0], block_size):
            for c in range(0, RI.shape[1], block_size):
                if ((r // block_size) + (c // block_size)) % 2 == 0:
                    checker[r:r+block_size, c:c+block_size, :] = RI[r:r+block_size, c:c+block_size, :3]
                else:
                    checker[r:r+block_size, c:c+block_size, :] = F[r:r+block_size, c:c+block_size, :]

        checker_fname = f"{save_dir}/{PurePath(moving_file).stem}_checker.png"
        print('Saving checkerboard as ', checker_fname)
        io.imsave(checker_fname, checker.astype('uint8'))

    # Lines
    if lines is not None:
        line_spacing = lines
        for r in range(0,moving_img.GetSize()[1],line_spacing):
            for x in range(moving_img.GetSize()[0]):
                moving_img.SetPixel(x,r,(0,0,0))
        for c in range(0,moving_img.GetSize()[0],line_spacing):
            for y in range(moving_img.GetSize()[1]):
                moving_img.SetPixel(c,y,(0,0,0))
            
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        resampled_image = resampler.Execute(moving_img)
        
        RIL=sitk.GetArrayFromImage(resampled_image)
        if len(RIL.shape) == 3 and RIL.shape[0] == 3:  
            RIL = np.transpose(RIL, (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
        RIL = RIL/np.max(RI)*255
        RIL=RIL.astype('uint8')

        lines_fname = f"{save_dir}/{PurePath(moving_file).stem}_lines.png"
        print('Saving image with deformed lines as', lines_fname)
        io.imsave(lines_fname, RIL.astype('uint8'))


ct_file = "/home/sagar/projects/GHOST/data/raw/45_ct.tif"
histo_file = "/home/sagar/projects/GHOST/data/raw/45_histo.tif"
register_2D_image(fixed_file=ct_file, moving_file=histo_file, fixed_mask=None, moving_mask=None, 
                      number_of_levels=3, max_iter=100, checkers=200, lines=100,
                      processed_dir=None, copy_originals=True)



'''
    plt.figure(figsize=(19,19))
    ax=plt.subplot(221)
    ax.imshow(RI.astype('uint8'))
    ax.set_title('deformed image')
    ax.axis('off')
    ax=plt.subplot(222)
    ax.imshow(checker.astype('uint8'))
    ax.set_title('checker board')
    ax.axis('off')
    ax=plt.subplot(223)
    ax.imshow(RIL.astype('uint8'))
    ax.set_title('deformed grid')
    ax.axis('off')
    ax=plt.subplot(224)
    mp=ax.imshow(jacobian,cmap='bwr')
    ax.set_title('Jacobian')
    plt.colorbar(mp)
    ax.axis('off')
    plt.show()


    '''