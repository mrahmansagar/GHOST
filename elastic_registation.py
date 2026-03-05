import argparse
from src.reg_pipeline import run_registration_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the image registration pipeline.",
                                     epilog="Example usage: python elastic_registration.py --fixed path/to/fixed.tif --moving path/to/moving.tif --output_dir path/to/output --create_checkers --create_lines")
    parser.add_argument("--fixed", 
                        type=str, 
                        required=True, 
                        metavar="<fixed_image>", 
                        help="Path to the fixed image file.")
    
    parser.add_argument("--moving", 
                        type=str, 
                        required=True, 
                        metavar="<moving_image>", 
                        help="Path to the moving image file.")
    
    parser.add_argument("--output_dir", 
                        type=str, 
                        default="./registration_output", 
                        help="Directory to save outputs.")
    
    parser.add_argument("--copy_originals", 
                        action='store_true', 
                        default=False, 
                        help="Whether to copy original image files.")
    
    parser.add_argument("--create_checkers", 
                        action='store_true', 
                        default=False, 
                        help="Whether to create checkerboard overlays.")
    
    parser.add_argument("--create_lines", 
                        action='store_true', 
                        default=False, 
                        help="Whether to create deformed grid images.")
    
    args = parser.parse_args()
    
    run_registration_pipeline(
        fixed_file=args.fixed,
        moving_file=args.moving,
        processed_dir=args.output_dir,
        copy_originals=args.copy_originals,
        create_checkers=args.create_checkers,
        create_lines=args.create_lines
    )

