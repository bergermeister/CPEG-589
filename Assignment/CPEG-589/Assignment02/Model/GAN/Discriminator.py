import torch.nn as nn

# Generative Adversarial Network (GAN) Discriminator
class Discriminator( nn.Module ):
   def __init__( self, channels ):
      super( ).__init__( )

      self.module = nn.Sequential( nn.Linear( channels * 1024, 512 ),
                                   nn.LeakyReLU( 0.2 ), 
                                   nn.Linear( 512, 256 ), 
                                   nn.LeakyReLU( 0.2 ),
                                   nn.Linear( 256, 1 ),
                                   nn.Sigmoid( ) )

   def forward( self, x ):
      return( self.module( x ) )
      
