from tfgans import utils

def load_data(
        src_dir: str,
        tar_dir: str,
        target_size: tuple = None,
        color_mode: str = 'rgb',
        interpolation='nearest',
        keep_aspect_ratio=False
):

    """
    Loads, resizes (if specified), and scales paired source and target image datasets.

    Args:
        src_dir (str): Path to source images folder.
        tar_dir (str): Path to target images folder.
        target_size (tuple or None): Dimensions (height, width) to resize images. Default is None.
        color_mode (str): Mode for loading images ('rgb' or 'grayscale'). Default is 'rgb'.
        min_pix_val (int): Minimum pixel intensity. Default is 0.
        max_pix_val (int): Maximum pixel intensity. Default is 255.
        final_activation (str): Activation range to scale to. Default is 'tanh' (maps to [-1, 1]).
        **kwargs: Additional parameters passed to utils.load_images_in_shape.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray): Scaled source and target data arrays.
    """

    load_kwargs = {
        'target_size': target_size,
        'color_mode': color_mode,
        'interpolation': interpolation,
        'keep_aspect_ratio': keep_aspect_ratio
   }

    src_data = utils.load_images_in_shape(src_dir, **load_kwargs)
    tar_data = utils.load_images_in_shape(tar_dir, **load_kwargs)

    return src_data, tar_data


def scale_paired_data(src_data, tar_data, min_pix_val=0, max_pix_val=255, final_activation="tanh"):
    """
    Scales paired source and target datasets to match model activation requirements.

    Args:
        src_data (numpy.ndarray): Source dataset.
        tar_data (numpy.ndarray): Target dataset.
        min_pix_val (int): Minimum pixel intensity. Default is 0.
        max_pix_val (int): Maximum pixel intensity. Default is 255.
        final_activation (str): Target activation range ('tanh' maps to [-1, 1]).

    Returns:
        tuple (numpy.ndarray, numpy.ndarray): Scaled source and target image arrays.
    """
    print(f"[*] Scaling datasets using activation: '{final_activation}'")
    src_scaled = utils.scale_data(
        src_data,
        min_pix_val=min_pix_val,
        max_pix_val=max_pix_val,
        final_activation=final_activation,
    )
    tar_scaled = utils.scale_data(
        tar_data,
        min_pix_val=min_pix_val,
        max_pix_val=max_pix_val,
        final_activation=final_activation,
    )

    return src_scaled, tar_scaled


