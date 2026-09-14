# GHOST
Generative Histology using Optimal Style Transfer. 


# Elastic Registration CLI

A small CLI wrapper to run the GHOST elastic image registration pipeline.  
The script parses command-line arguments and calls the registration pipeline implemented in [src/reg_pipeline.py](src/reg_pipeline.py).

Script: [elastic_registation.py](elastic_registation.py)

## Usage

Basic invocation:
```bash
python3 elastic_registation.py --fixed path/to/fixed.tif --moving path/to/moving.tif
```

Full example:
```bash
python elastic_registation.py \
  --fixed data/processed/colon/fixed.tif \
  --moving data/processed/colon/moving.tif \
  --output_dir ./registration_output \
  --copy_originals \
  --max_iter 200 \
  --create_checkers 100 \
  --create_lines 100
```

## CLI Arguments

- `--fixed FILE` (required): Path to the fixed (reference) image.
- `--moving FILE` (required): Path to the moving image to be registered.
- `--output_dir DIR`: Directory to save registered images and visualizations. Default: `./registration_output`.
- `--copy_originals`: Flag; if present, copies original input files into `--output_dir`.
- `--max_iter N`: Maximum iterations for the optimizer (LBFGSB). Default: `100`.
- `--create_checkers BLOCK_SIZE`: If >0, generates a checkerboard image for alignment verification. Default: `100`.
- `--create_lines LINE_SPACING`: If >0, generates a deformed-grid/line visualization. Default: `100`.