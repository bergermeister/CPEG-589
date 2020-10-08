import torch.nn as nn

# Deep Convolutional (DC) Generative Adversarial Network (GAN) Generator
class Generator( nn.Module ):
   def __init__( self, channels ):
      super( ).__init__( )
      # Filters [1024, 512, 256]
      # Input_dim = 100
      # Output_dim = C (number of channels)
      self.module = nn.Sequential(
         # Z latent vector 100
         ConvLayer( in_channels = 100, out_channels = 1024, kernel_size = 4, stride = 1, padding = 0, num_features = 1024 ),
         
         # State (1024x4x4)
         ConvLayer( in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, num_features = 512 ),

         # State (512x8x8)
         ConvLayer( in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, num_features = 256 ),

         # State (256x16x16)
         nn.ConvTranspose2d( in_channels = 256, out_channels = channels, kernel_size = 4, stride = 2, padding = 1 )
      ) # output of module --> Image (Cx32x32)

      self.output = nn.Tanh( )

   def forward( self, x ):
      x = self.module( x )
      return( self.output( x ) )

def ConvLayer( in_channels, out_channels, kernel_size, stride, padding, num_features ):
   return( nn.Sequential( nn.ConvTranspose2d( in_channels = in_channels, out_channels = out_channels, 
                                              kernel_size = kernel_size, stride = stride, padding = padding ),
                          nn.BatchNorm2d( num_features = num_features ),
                          nn.ReLU( True ) ) )

