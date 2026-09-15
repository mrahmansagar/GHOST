from tfgans.pix2pix import models
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
    dis = models.build_discriminator(
        src_shape=source_shape,
        tar_shape=target_shape,
        optimizer=dis_opt, 
        lr=dis_lr,
        beta1=dis_beta_1,
        loss=dis_loss,
        loss_weights=dis_loss_weights,
        metrics=dis_metrics)

    # Initialize the Generator
    gen = models.build_generator(
        input_shape=source_shape,
        output_channel=gen_output_channel)

    # Initialize the Conditional GAN
    cgan = models.build_pix2pix(
        generator=gen,
        discriminator=dis,
        opt=cgan_opt,
        lr=cgan_lr,
        beta1=cgan_beta_1,
        loss=cgan_loss,
        loss_weights=cgan_loss_weights)
     
    return dis, gen, cgan

