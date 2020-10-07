import argparse
import os
import torch

class Configuration( object ):
   def __init__( self ):
      parser = argparse.ArgumentParser( description = "CPEG-589 Implementation of GAN models using PyTorch" )
       
      parser.add_argument('--model', type=str, default='DCGAN', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
      parser.add_argument('--is_train', type=str, default='True')
      parser.add_argument('--dataroot', required=True, help='path to dataset')
      parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                         help='The name of dataset')
      parser.add_argument('--download', type=str, default='False')
      parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
      parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
      parser.add_argument('--cuda',  type=str, default='False', help='Availability of cuda')
      
      parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
      parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
      parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')

      self.Arguments = parser.parse_args( )

      # Verify --epoch
      try:
         assert self.Arguments.epochs >= 1
      except:
         print( 'Number of epochs must be larger than or equal to one' )

      # Verify --batch_size
      try:
         assert self.Arguments.batch_size >= 1
      except:
         print( 'Batch size must be larger than or equal to one' )

      if( ( self.Arguments.dataset == 'cifar' ) or ( self.Arguments.dataset == 'stl10' ) ):
         self.Arguments.channels = 3
      else:
         self.Arguments.channels = 1

      # Verify --cuda
      if( self.Arguments.cuda == 'True' ):
         if( torch.cuda.is_available( ) == True ):
            self.Arguments.cuda = True 
         else:
            self.Arguments.cuda = False
            print( 'Cuda is not available' )
      else:
         self.Arguments.cuda = False
