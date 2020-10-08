import torch
from Model.GAN.Network import Network as GAN
from Model.GAN.DC.Generator import Generator
from Model.GAN.DC.Discriminator import Discriminator

# Deep Convolutional (DC) Generative Adversarial Network (GAN)
class Network( GAN ):
   def __init__( self, args ):
      print("DCGAN model initalization.")

      # Create Generator and Discriminator
      Gen = Generator( args.channels )
      Dis = Discriminator( args.channels )

      # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
      d_optimizer = torch.optim.Adam( Dis.parameters( ), lr = 0.0002, betas = ( 0.5, 0.999 ) )
      g_optimizer = torch.optim.Adam( Gen.parameters( ), lr = 0.0002, betas = ( 0.5, 0.999 ) )
      super( ).__init__( Gen, Dis, g_optimizer, d_optimizer, args )
