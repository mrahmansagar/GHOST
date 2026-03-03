from pathlib import Path
import shutil
from skimage import io
import SimpleITK as sitk

# Import our new modular functions
from src.registration import compute_bspline_transform, apply_transform, preprocess_for_registration
from src.reg_utils import create_checkerboard_overlay, create_deformed_grid

def run_registration_pipeline(
    fixed_file: str, 
    moving_file: str, 
    processed_dir: str = "./registration_output", 
    copy_originals: bool = False,
    create_checkers: bool = False,
    create_lines: bool = False 
):
    """
    Orchestrates the loading, registration, and saving of images.
    Args:
        fixed_file: Path to the fixed image file.
        moving_file: Path to the moving image file.
        processed_dir: Directory where outputs will be saved.
        create_checkers: Whether to create and save checkerboard overlays.
        create_lines: Whether to create and save deformed grid images.
    """

    # 1. Load Original Data and Preprocess for Registration
    fixed_image = io.imread(fixed_file).astype('float')
    moving_image = io.imread(moving_file).astype('float')

    # 2. Preprocess for Registration (Averaging channels and normalizing)
    fixed_image_sitk, moving_image_sitk = preprocess_for_registration(fixed_image, moving_image)

    # 3. Compute Transform
    transform = compute_bspline_transform(fixed_image_sitk, moving_image_sitk)

    # 4. Load Original Data as SimpleITK Images for Applying Transform (to preserve metadata)
    fixed_sitk = sitk.ReadImage(fixed_file)
    moving_sitk = sitk.ReadImage(moving_file)
    
    # 5. Apply Transform
    registered_array = apply_transform(fixed_sitk, moving_sitk, transform)
    
    # 4. Save Outputs (Fixing the directory bug!)
    save_dir = Path(processed_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_originals:
    # Fixed File
        shutil.copy2(fixed_file, save_dir)

        # Moving file
        shutil.copy2(moving_file, save_dir)  


    moving_stem = Path(moving_file).stem
    reg_fname = save_dir / f"{moving_stem}_registered.tif"
    
    io.imsave(reg_fname, registered_array)
    print(f"Saved: {reg_fname}")
    
    # 5. Optional Visualization
    if create_checkers:
        fixed_array = io.imread(fixed_file).astype('float')
        checker_img = create_checkerboard_overlay(fixed_array, registered_array, block_size=150)
        io.imsave(save_dir / f"{moving_stem}_checker.png", checker_img)

    # 6. Optional: Create and save deformed grid (not implemented here, but can be added similarly)
    if create_lines:
        grid_img = create_deformed_grid(fixed_sitk, moving_sitk, transform, line_spacing=200)
        io.imsave(save_dir / f"{moving_stem}_lines.png", grid_img)

