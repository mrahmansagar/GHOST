from tfgans.pix2pix import models as pix2pix_models
from tfgans.cycleGAN import models as cycle_models
from keras import Model

from keras.optimizers import Adam

def create_pix2pix_model(

    source_shape: tuple,
    target_shape: tuple,
    dis_opt = Adam,
    dis_lr: float = 0.0002,
    dis_beta_1: float = 0.5,
    dis_loss: str = 'binary_crossentropy',
    dis_loss_weights: list = [0.5],
    dis_metrics: list = ['accuracy'],
    gen_output_channel: int = None,
    cgan_opt = Adam,
    cgan_lr: float = 0.0002,
    cgan_beta_1: float = 0.5,
    cgan_loss: list = ['binary_crossentropy', 'mae'],
    cgan_loss_weights: list = [1, 100]
):
    """
    Creates a pix2pix model with specified configurations.
    
    Args:
        source_shape: Shape of the source images (input to the generator).
        target_shape: Shape of the target images (output from the generator).
        dis_opt: Optimizer for the discriminator.
        dis_lr: Learning rate for the discriminator.
        dis_beta_1: Beta_1 parameter for the discriminator optimizer.
        dis_loss: Loss function for the discriminator.
        dis_loss_weights: Loss weights for the discriminator.
        dis_metrics: Metrics for evaluating the discriminator.
        gen_output_channel: Number of output channels for the generator.
        cgan_opt: Optimizer for the conditional GAN.
        cgan_lr: Learning rate for the conditional GAN.
        cgan_beta_1: Beta_1 parameter for the conditional GAN optimizer.
        cgan_loss: Loss functions for the conditional GAN.
        cgan_loss_weights: Loss weights for the conditional GAN.

    Returns:
        A compiled pix2pix model ready for training.
    
    
    Initializes and links the Generator, Discriminatiorand Conditional GAN models for the pix2pix architecture.
    """

    # Initialize the Discriminator
    dis = pix2pix_models.build_discriminator(
        src_shape=source_shape,
        tar_shape=target_shape,
        optimizer=dis_opt, 
        lr=dis_lr,
        beta1=dis_beta_1,
        loss=dis_loss,
        loss_weights=dis_loss_weights,
        metrics=dis_metrics)

    # Initialize the Generator
    gen = pix2pix_models.build_generator(
        input_shape=source_shape,
        output_channel=gen_output_channel)

    # Initialize the Conditional GAN
    cgan = pix2pix_models.build_pix2pix(
        generator=gen,
        discriminator=dis,
        opt=cgan_opt,
        lr=cgan_lr,
        beta1=cgan_beta_1,
        loss=cgan_loss,
        loss_weights=cgan_loss_weights)
     
    return dis, gen, cgan


def create_cycleGAN_model(
    domainA_shape: tuple,
    domainB_shape: tuple,
    disA_opt = Adam,
    disB_opt = Adam,
    disA_lr: float = 0.0002,
    disB_lr: float = 0.0002,
    disA_beta_1: float = 0.5,
    disB_beta_1: float = 0.5,
    disA_loss: str = 'mse',
    disB_loss: str = 'mse',
    disA_loss_weights: list = [0.5],
    disB_loss_weights: list = [0.5],
    genA2B_residual_blocks: int = 9,
    genB2A_residual_blocks: int = 9,
    cycleA2B_opt = Adam,
    cycleB2A_opt = Adam,
    cycleA2B_lr: float = 0.0002,
    cycleB2A_lr: float = 0.0002,
    cycleA2B_beta_1: float = 0.5,
    cycleB2A_beta_1: float = 0.5,
    cycleA2B_loss: list = ['mse', 'mae', 'mae', 'mae'],
    cycleB2A_loss: list = ['mse', 'mae', 'mae', 'mae'],
    cycleA2B_loss_weights: list = [1, 5, 10, 10],
    cycleB2A_loss_weights: list = [1, 5, 10, 10]
):

    """
    Creates a cycleGAN model with specified configurations.

    Args:
        domainA_shape: Shape of the images in domain A.
        domainB_shape: Shape of the images in domain B.
        disA_opt: Optimizer for the discriminator of domain A.
        disB_opt: Optimizer for the discriminator of domain B.
        disA_lr: Learning rate for the discriminator of domain A.
        disB_lr: Learning rate for the discriminator of domain B.
        disA_beta_1: Beta_1 parameter for the discriminator of domain A.
        disB_beta_1: Beta_1 parameter for the discriminator of domain B.
        disA_loss: Loss function for the discriminator of domain A.
        disB_loss: Loss function for the discriminator of domain B.
        disA_loss_weights: Loss weights for the discriminator of domain A.
        disB_loss_weights: Loss weights for the discriminator of domain B.
        genA2B_residual_blocks: Number of residual blocks in the generator from domain A to domain B.
        genB2A_residual_blocks: Number of residual blocks in the generator from domain B to domain A.
        cycleA2B_opt: Optimizer for the cycleGAN from domain A to domain B.
        cycleB2A_opt: Optimizer for the cycleGAN from domain B to domain A.
        cycleA2B_lr: Learning rate for the cycleGAN from domain A to domain B.
        cycleB2A_lr: Learning rate for the cycleGAN from domain B to domain A.
        cycleA2B_beta_1: Beta_1 parameter for the cycleGAN from domain A to domain B.
        cycleB2A_beta_1: Beta_1 parameter for the cycleGAN from domain B to domain A.
        cycleA2B_loss: Loss functions for the cycleGAN from domain A to domain B.
        cycleB2A_loss: Loss functions for the cycleGAN from domain B to domain A.
        cycleA2B_loss_weights: Loss weights for the cycleGAN from domain A to domain B.
        cycleB2A_loss_weights: Loss weights for the cycleGAN from domain B to domain A.
    """ 

    # Initialize the Discriminators
    disA = cycle_models.build_discriminator(
        input_shape=domainA_shape,
        opt=disA_opt, 
        lr=disA_lr,
        beta1=disA_beta_1,
        loss=disA_loss,
        loss_weights=disA_loss_weights)

    disB = cycle_models.build_discriminator(
        input_shape=domainB_shape,
        opt=disB_opt, 
        lr=disB_lr,
        beta1=disB_beta_1,
        loss=disB_loss,
        loss_weights=disB_loss_weights)

    # Initialize the Generators
    genA2B = cycle_models.build_generator(
        input_shape=domainA_shape,
        sizeof_resnet_block=genA2B_residual_blocks)

    genB2A = cycle_models.build_generator(
        input_shape=domainB_shape,
        sizeof_resnet_block=genB2A_residual_blocks)

    # Initialize the CycleGANs
    cycleGAN_A2B = cycle_models.build_cycleGAN(
        gen1=genA2B,
        dis=disB,
        gen2=genB2A,
        input_shape=domainA_shape,
        opt=cycleA2B_opt,
        lr=cycleA2B_lr,
        beta1=cycleA2B_beta_1,
        loss=cycleA2B_loss,
        loss_weights=cycleA2B_loss_weights)

    cycleGAN_B2A = cycle_models.build_cycleGAN(
        gen1=genB2A,
        dis=disA,
        gen2=genA2B,
        input_shape=domainB_shape,
        opt=cycleB2A_opt,
        lr=cycleB2A_lr,
        beta1=cycleB2A_beta_1,
        loss=cycleB2A_loss,
        loss_weights=cycleB2A_loss_weights)

    return disA, disB, genA2B, genB2A, cycleGAN_A2B, cycleGAN_B2A 

    