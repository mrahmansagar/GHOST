import argparse
import keras
from keras.optimizers import Adam

from tfgans.pix2pix import models
from src.models import create_pix2pix_model
from src.dataloader import load_data, scale_paired_data

def parse_args():

    parser = argparse.ArgumentParser(
            description="CLI tool to train Pix2Pix GAN using tfgans",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=(
                "Example usage:\n"
                "python3 train.py --src_dir path/to/ct_images --tar_dir path/to/histo_images "
                "--output_dir ./training_output --epochs 100"
            )
        )
    # Directory paths
    parser.add_argument("--src_dir", 
                        type=str, 
                        required=True, 
                        help="Directory containing source images")

    parser.add_argument("--tar_dir", 
                        type=str, 
                        required=True, 
                        help="Directory containing target images")

    # Image resizing and formatting
    parser.add_argument("--img_height", 
                        type=int, 
                        default=None, 
                        help="Target image height")
    
    parser.add_argument("--img_width", 
                        type=int, 
                        default=None, 
                        help="Target image width")
    
    parser.add_argument("--color_mode", 
                        type=str, 
                        default="rgb", 
                        choices=["rgb", "grayscale"], 
                        help="Color mode for images")


    # Model hyperparameters
    parser.add_argument("--dis_optimizer", 
                        type=keras.optimizers.Optimizer, 
                        default=Adam, 
                        help="Optimizer for the Discriminator")

    parser.add_argument("--dis_lr", 
                        type=float, 
                        default=0.0002, 
                        help="Learning rate for the Discriminator optimizer")

    parser.add_argument("--dis_beta1", 
                        type=float, 
                        default=0.5,
                        help="Beta_1 parameter for the Discriminator optimizer")
    
    parser.add_argument("--dis_loss", 
                        type=str, 
                        default="binary_crossentropy", 
                        help="Loss function for the Discriminator")

    parser.add_argument("--dis_loss_weights", 
                        type=list, 
                        default=[0.5],
                        help="Loss weights for the Discriminator")

    parser.add_argument("--dis_metrics", 
                        type=list, 
                        default=["accuracy"],
                        help="Metrics for evaluating the Discriminator")
    
    # Generator and CGAN hyperparameters
    parser.add_argument("--gen_output_channel", 
                        type=int, 
                        default=None,
                        help="Number of output channels for the Generator")

    parser.add_argument("--cgan_optimizer", 
                        type=keras.optimizers.Optimizer, 
                        default=Adam,  
                        help="Optimizer for the CGAN")

    parser.add_argument("--cgan_lr", 
                        type=float, 
                        default=0.0002, 
                        help="Learning rate for the CGAN optimizer")
    
    parser.add_argument("--cgan_beta1", 
                        type=float, 
                        default=0.5, 
                        help="Adam optimizer beta1 parameter")

    parser.add_argument("--cgan_loss", 
                        type=list, 
                        default=["binary_crossentropy", "mae"],
                        help="Loss functions for the CGAN")

    parser.add_argument("--cgan_loss_weights", 
                        type=list, 
                        default=[1, 100.0], 
                        help="Weight multiplier for Generator L1 MAE loss")
    
    # Training configurations
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1, 
                        help="Batch size for training")
    
    parser.add_argument("--epochs", 
                        type=int, 
                        default=10, 
                        help="Total number of epochs")
    
    parser.add_argument("--summary_interval", 
                        type=int, 
                        default=10, 
                        help="Interval (epochs) to evaluate & save samples")
    
    parser.add_argument("--run_name", 
                        type=str, 
                        default="pix2pix", 
                        help="Identifier prefix for checkpoints and logs")

    

    return parser.parse_args()

def main():
    args = parse_args()

    # Determine image dimensions
    target_size = None
    if args.img_height is not None and args.img_width is not None:
        target_size = (args.img_height, args.img_width)

    # 1. Load data
    src_data, tar_data = load_data(
        src_dir=args.src_dir,
        tar_dir=args.tar_dir,
        target_size=target_size,
        color_mode=args.color_mode,
    )

    # 2. Scale data
    src_data, tar_data = scale_paired_data(
        src_data=src_data,
        tar_data=tar_data,
        min_pix_val=0,
        max_pix_val=255,
        final_activation="tanh",
    )

    src_shape = src_data.shape[1:]
    tar_shape = tar_data.shape[1:]
    print(f"[*] Prepared Shapes -> Source: {src_shape} | Target: {tar_shape}")

    # 3. Build model architecture
    dis, gen, cgan = create_pix2pix_model(
        source_shape=src_shape,
        target_shape=tar_shape,
        dis_opt = args.dis_optimizer,
        dis_lr=args.dis_lr,
        dis_beta_1=args.dis_beta1,
        dis_loss=args.dis_loss,
        dis_loss_weights=args.dis_loss_weights,
        dis_metrics=args.dis_metrics,
        gen_output_channel=args.gen_output_channel,
        cgan_opt = args.cgan_optimizer,
        cgan_lr=args.cgan_lr,
        cgan_beta_1=args.cgan_beta1,
        cgan_loss=args.cgan_loss,
        cgan_loss_weights=args.cgan_loss_weights,
    )

    # 4. Run training
    print("[*] Starting Pix2Pix training loop...")
    models.train_pix2pix(
        gen=gen,
        dis=dis,
        cgan=cgan,
        src_data=src_data,
        tar_data=tar_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        summary_interval=args.summary_interval,
        name=args.run_name,
    )

if __name__ == "__main__":
    main()
