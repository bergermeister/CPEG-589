import torch.nn as nn

# Generative Adversarial Network (GAN) Generator
class Generator( nn.Module ):
   def __init__( self, channels ):
      super( ).__init__( )

      self.module = nn.Sequential( nn.Linear( 100, 256 ),
                                   nn.LeakyReLU( 0.2 ),
                                   nn.Linear( 256, 512 ),
                                   nn.LeakyReLU( 0.2 ),
                                   nn.Linear( 512, 1024 ),
                                   nn.LeakyReLU( 0.2 ),
                                   nn.Tanh( ) )

   def forward( self, x ):
      return( self.module( x ) )
