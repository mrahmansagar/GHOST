import argparse
import keras
from keras.optimizers import Adam

from tfgans.cycleGAN import models
from src.models import create_cycleGAN_model
from src.dataloader import load_data, scale_paired_data

def parse_args():

    parser = argparse.ArgumentParser(
            description="CLI tool to train CycleGAN using tfgans",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=(
                "Example usage:\n"
                "python3 train_cycleGAN.py --domainA_dir path/to/ct_images --domainB_dir path/to/histo_images "
                "--output_dir ./training_output --epochs 10" 
            )
        )
    # Directory paths
    parser.add_argument("--domainA_dir", 
                        type=str, 
                        required=True, 
                        help="Directory containing domain A images")

    parser.add_argument("--domainB_dir", 
                        type=str, 
                        required=True, 
                        help="Directory containing domain B images")

    # Image resizing and formatting
    parser.add_argument("--img_height", 
                        type=int, 
                        default=None, 
                        help="Image height")
    
    parser.add_argument("--img_width", 
                        type=int, 
                        default=None, 
                        help="Image width")
    
    parser.add_argument("--color_mode", 
                        type=str, 
                        default="rgb", 
                        choices=["rgb", "grayscale"], 
                        help="Color mode for images")


    # Model hyperparameters
    parser.add_argument("--disA_optimizer", 
                        type=keras.optimizers.Optimizer, 
                        default=Adam, 
                        help="Optimizer for the Discriminator A")

    # Model hyperparameters
    parser.add_argument("--disB_optimizer", 
                        type=keras.optimizers.Optimizer, 
                        default=Adam, 
                        help="Optimizer for the Discriminator B")

    parser.add_argument("--disA_lr", 
                        type=float, 
                        default=0.0002, 
                        help="Learning rate for the Discriminator A optimizer")

    parser.add_argument("--disB_lr", 
                        type=float, 
                        default=0.0002, 
                        help="Learning rate for the Discriminator B optimizer") 

    parser.add_argument("--disA_beta1",
                        type=float, 
                        default=0.5,
                        help="Beta_1 parameter for the Discriminator A optimizer")        

    parser.add_argument("--disB_beta1",
                        type=float, 
                        default=0.5,
                        help="Beta_1 parameter for the Discriminator B optimizer")  

    parser.add_argument("--disA_loss",
                        type=str,
                        default="mse", 
                        help="Loss function for the Discriminator A")

    parser.add_argument("--disB_loss",
                        type=str,
                        default="mse", 
                        help="Loss function for the Discriminator B")   

    parser.add_argument("--disA_loss_weights",
                        type=list,
                        default=[0.5],
                        help="Loss weights for the Discriminator A")

    parser.add_argument("--disB_loss_weights",
                        type=list,
                        default=[0.5],
                        help="Loss weights for the Discriminator B")    

    parser.add_argument("--genA2B_residual_blocks",
                        type=int,
                        default=9,
                        help="Number of residual blocks in the generator from domain A to domain B")    

    parser.add_argument("--genB2A_residual_blocks",
                        type=int,
                        default=9,
                        help="Number of residual blocks in the generator from domain B to domain A")    

    parser.add_argument("--cycleA2B_optimizer",
                        type=keras.optimizers.Optimizer,
                        default=Adam,
                        help="Optimizer for the cycleGAN from domain A to domain B")    

    parser.add_argument("--cycleB2A_optimizer",
                        type=keras.optimizers.Optimizer,
                        default=Adam,
                        help="Optimizer for the cycleGAN from domain B to domain A")

    parser.add_argument("--cycleA2B_lr",
                        type=float,
                        default=0.0002,
                        help="Learning rate for the cycleGAN from domain A to domain B")

    parser.add_argument("--cycleB2A_lr",
                        type=float,
                        default=0.0002,
                        help="Learning rate for the cycleGAN from domain B to domain A")

    parser.add_argument("--cycleA2B_beta1",
                        type=float,
                        default=0.5,
                        help="Beta_1 parameter for the cycleGAN from domain A to domain B")

    parser.add_argument("--cycleB2A_beta1",
                        type=float,
                        default=0.5,
                        help="Beta_1 parameter for the cycleGAN from domain B to domain A")

    parser.add_argument("--cycleA2B_loss",
                        type=list,
                        default=['mse', 'mae', 'mae', 'mae'],
                        help="Loss functions for the cycleGAN from domain A to domain B")

    parser.add_argument("--cycleB2A_loss",
                        type=list,
                        default=['mse', 'mae', 'mae', 'mae'],
                        help="Loss functions for the cycleGAN from domain B to domain A")

    parser.add_argument("--cycleA2B_loss_weights",
                        type=list,
                        default=[1, 5, 10, 10],
                        help="Loss weights for the cycleGAN from domain A to domain B")

    parser.add_argument("--cycleB2A_loss_weights",
                        type=list,
                        default=[1, 5, 10, 10],
                        help="Loss weights for the cycleGAN from domain B to domain A")

    # Training parameters
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1, 
                        help="Batch size for training")

    parser.add_argument("--epochs",
                        type=int, 
                        default=10, 
                        help="Number of epochs for training")

    parser.add_argument("--summary_interval",
                        type=int, 
                        default=1, 
                        help="Interval (in epochs) for printing training summaries")

    parser.add_argument("--run_name",
                        type=str, 
                        default="cycleGAN", 
                        help="Name for the  training run (used for saving models and logs)")

    parser.add_argument("--DomainA_name",
                        type=str,
                        default="A",
                        help="Name for domain A")

    parser.add_argument("--DomainB_name",
                        type=str,
                        default="B",
                        help="Name for domain B")
    


    return parser.parse_args()

def main():
    args = parse_args()

    target_size = None
    if args.img_height is not None and args.img_width is not None:
        target_size = (args.img_height, args.img_width)

    nameA2B = f"Gen_{args.DomainA_name}2{args.DomainB_name}"
    nameB2A = f"Gen_{args.DomainB_name}2{args.DomainA_name}"

    # Load the data from domain A and domain B directories
    domainA_data, domainB_data = load_data(
        src_dir = args.domainA_dir,
        tar_dir = args.domainB_dir,
        target_size=target_size,
        color_mode=args.color_mode)    
    

    # Scale the data to match the model's expected input range
    domainA_data, domainB_data = scale_paired_data(
        src_data=domainA_data,
        tar_data=domainB_data,
        min_pix_val=0,
        max_pix_val=255,
        final_activation="tanh")


    domainA_shape = domainA_data.shape[1:]
    domainB_shape = domainB_data.shape[1:]
    print(f"[*] Prepared Shapes -> Domain A: {domainA_shape} | Domain B: {domainB_shape}")

    # Create CycleGAN model
    disA, disB, gneA2B, genB2A, cycleGAN_A2B, cycleGAN_B2A = create_cycleGAN_model(
        domainA_shape=domainA_shape,
        domainB_shape=domainB_shape,
        disA_opt=args.disA_optimizer,
        disB_opt=args.disB_optimizer,
        disA_lr=args.disA_lr,
        disB_lr=args.disB_lr,
        disA_beta_1=args.disA_beta1,
        disB_beta_1=args.disB_beta1,
        disA_loss=args.disA_loss,
        disB_loss=args.disB_loss,
        disA_loss_weights=args.disA_loss_weights,
        disB_loss_weights=args.disB_loss_weights,
        genA2B_residual_blocks=args.genA2B_residual_blocks,
        genB2A_residual_blocks=args.genB2A_residual_blocks,
        cycleA2B_opt=args.cycleA2B_optimizer,
        cycleB2A_opt=args.cycleB2A_optimizer,
        cycleA2B_lr=args.cycleA2B_lr,
        cycleB2A_lr=args.cycleB2A_lr,
        cycleA2B_beta_1=args.cycleA2B_beta1,
        cycleB2A_beta_1=args.cycleB2A_beta1,
        cycleA2B_loss=args.cycleA2B_loss,
        cycleB2A_loss=args.cycleB2A_loss,
        cycleA2B_loss_weights=args.cycleA2B_loss_weights,
        cycleB2A_loss_weights=args.cycleB2A_loss_weights
    )

    # Run the training loop for CycleGAN
    print("[*] Starting CycleGAN training loop...")
    models.train_cycleGAN(
        disA = disA,
        disB = disB,
        genA2B = gneA2B,
        genB2A = genB2A,
        cganA2B = cycleGAN_A2B,
        cganB2A = cycleGAN_B2A,
        dataA = domainA_data,
        dataB = domainB_data,
        batch_size = args.batch_size,
        epochs = args.epochs,
        summary_interval = args.summary_interval,
        name = args.run_name,
        nameA2B = nameA2B,
        nameB2A = nameB2A
    )

if __name__ == "__main__":
    main()

