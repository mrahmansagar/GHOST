import argparse
import sys
from src.reg_pipeline import run_registration_pipeline

def main():
    """
    Entry point for the GHOST elastic registration CLI.
    Parses command line arguments and executes the registration pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run the GHOST elastic image registration pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example usage:\n"
            "python elastic_registration.py --fixed path/to/fixed.tif "
            "--moving path/to/moving.tif --output_dir ./results --create_checkers"
        )
    )

    # --- Path Arguments ---
    parser.add_argument(
        "--fixed", 
        type=str, 
        required=True, 
        metavar="FILE", 
        help="Path to the fixed (reference) image file."
    )
    
    parser.add_argument(
        "--moving", 
        type=str, 
        required=True, 
        metavar="FILE", 
        help="Path to the moving image file to be registered."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./registration_output", 
        help="Directory where registered images and visualizations will be saved."
    )
    
    # --- Flags (Booleans) ---
    parser.add_argument(
        "--copy_originals", 
        action='store_true', 
        help="If set, copies the original input files into the output directory."
    )
    
    parser.add_argument(
        "--create_checkers", 
        action='store_true', 
        help="If set, generates a checkerboard visualization for alignment verification."
    )
    
    parser.add_argument(
        "--create_lines", 
        action='store_true', 
        help="If set, generates an image showing the deformed grid lines."
    )
    
    # Parse the arguments from sys.argv
    args = parser.parse_args()
    
    # Execute the pipeline
    try:
        run_registration_pipeline(
            fixed_file=args.fixed,
            moving_file=args.moving,
            processed_dir=args.output_dir,
            copy_originals=args.copy_originals,
            create_checkers=args.create_checkers,
            create_lines=args.create_lines
        )
    except Exception as e:
        print(f"Error: Registration failed - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()