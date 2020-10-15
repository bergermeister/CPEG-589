import torch.nn as nn

# Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (GP) Discriminator
class Discriminator( nn.Module ):
   def __init__( self, channels ):
      super( ).__init__( )
      # Filters [256, 512, 1024]
      # Input_dim = channels (Cx64x64)
      # Output_dim = 1
      self.module = nn.Sequential(
         # Image (Cx32x32)
         ConvLayer( in_channels = channels, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, num_features = 256 ),

         # State (256x16x16)
         ConvLayer( in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, num_features = 512 ),

         # State (512x8x8)
         ConvLayer( in_channels = 512, out_channels = 1024, kernel_size = 4, stride = 2, padding = 1, num_features = 1024 )
      ) # outptut of main module --> State (1024x4x4)

      self.output = nn.Sequential(
         nn.Conv2d( in_channels = 1024, out_channels = 1, kernel_size = 4, stride = 1, padding = 0 ),
      )

   def forward( self, x ):
      x = self.module( x )
      return( self.output( x ) )

   def feature_extraction( self, x ):
      # Use discriminator for feature extraction then flatten to vector of 16384 features
      x = self.module( x )
      return( x.view( -1, 1024 * 4 * 4 ) )

def ConvLayer( in_channels, out_channels, kernel_size, stride, padding, num_features ):
   return( nn.Sequential( nn.Conv2d( in_channels = in_channels, out_channels = out_channels, 
                                     kernel_size = kernel_size, stride = stride, padding = padding ),
                          nn.InstanceNorm2d( num_features ),
                          nn.LeakyReLU( 0.2, inplace = True ) ) )

