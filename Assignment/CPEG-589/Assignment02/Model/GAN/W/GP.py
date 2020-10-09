import os
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torchvision import utils
from time import time
from Model.GAN.Network import Network as GAN
from Model.GAN.Network import GetInfiniteBatches
from Model.GAN.DC.Generator import Generator
from Model.GAN.DC.Discriminator import Discriminator

# Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (GP)
class GP( GAN ):
   def __init__( self, args ):
      print( "WGAN-GP Model Initialization" )

      # Create Generator and Discriminator
      Gen = Generator( args.channels )
      Dis = Discriminator( args.channels )
      
      # Create Optimizers using ADAM
      d_optimizer = torch.optim.Adam( Dis.parameters( ), lr = 1e-4, betas = ( 0.5, 0.999 ) )
      g_optimizer = torch.optim.Adam( Gen.parameters( ), lr = 1e-4, betas = ( 0.5, 0.999 ) )
      super( ).__init__( Gen, Dis, g_optimizer, d_optimizer, args )

      # Store iteration counts
      self.itersGenerator = args.generator_iters
      self.itersCritic = 5
      self.lambdaTerm = 10

   def Train( self, TrainLoader ):
      self.begin = time( )
      self.file = open( "inception_score_graph.txt", "w" )

      # Now batches are callable self.data.next()
      self.data = GetInfiniteBatches( TrainLoader )

      one = torch.tensor( 1, dtype = torch.float ) # torch.FloatTensor( [ 1 ] )
      mone = one * -1
      if( self.cudaEnable ):
         one = one.cuda( self.cudaIndex )
         mone = mone.cuda( self.cudaIndex )

      for iterG in range( self.itersGenerator ):
         # Requires grad, Generator requires_grad = False
         for p in self.D.parameters( ):
            p.requires_grad = True

         # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
         for iterD in range( self.itersCritic ):
            self.D.zero_grad( )

            images = self.data.__next__( )
            # Check for batch to have full batch_size
            if( images.size( )[ 0 ] != self.batchSize ):
               continue

            images = self.GetVariable( images )

            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            lossRealD = self.D( images )
            lossRealD = lossRealD.mean( )
            lossRealD.backward( mone )

            # Train with fake images
            z          = self.GetVariable( torch.randn( self.batchSize, 100, 1, 1 ) )
            imagesFake = self.G( z )
            lossFakeD  = self.D( imagesFake )
            lossFakeD  = lossFakeD.mean( )
            lossFakeD.backward( one )

            # Train with gradient penalty
            gradientPenalty = self.CalculateGradientPenalty( images.data, imagesFake.data )
            gradientPenalty.backward( )

            lossD = lossFakeD - lossRealD + gradientPenalty
            WassersteinD = lossRealD - lossFakeD
            self.optimizerD.step( )

         # Generator update
         for p in self.D.parameters( ):
            p.requires_grad = False  # to avoid computation

         self.G.zero_grad( )

         # Train generator
         # Compute loss with fake images
         z = self.GetVariable( torch.randn( self.batchSize, 100, 1, 1 ) )
         imagesFake = self.G(z)
         lossG = self.D( imagesFake )
         lossG = lossG.mean( ).mean( )
         lossG.backward( mone )
         costG = -lossG
         self.optimizerG.step( )

         # Saving model and sampling images every 1000th generator iterations
         if( ( iterG % 1000 ) == 0 ):
            self.save_model( )
            # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
            # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
            # This way Inception score is more correct since there are different generated examples from every class of Inception model
            # sample_list = []
            # for i in range(10):
            #    z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
            #    samples = self.G(z)
            #    sample_list.append(samples.data.cpu().numpy())
            #
            # # Flattening list of list into one list
            # new_sample_list = list(chain.from_iterable(sample_list))
            # print("Calculating Inception Score over 8k generated images")
            # # Feeding list of numpy arrays
            # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
            #                              resize=True, splits=10)
            if not os.path.exists( 'training_result_images/' ):
               os.makedirs( 'training_result_images/' )

            # Denormalize images and save them in grid 8x8
            z = self.GetVariable( torch.randn( 800, 100, 1, 1 ) )
            samples = self.G( z )
            samples = samples.mul( 0.5 ).add( 0.5 )
            samples = samples.data.cpu( )[ :64 ]
            grid = utils.make_grid( samples )
            utils.save_image( grid, 'training_result_images/img_generatori_iter_{}.png'.format( str( iterG ).zfill( 3 ) ) )

            # Testing
            elapsed = time( ) - self.begin
            print( "Generator iter: {}".format( iterG ) )
            print( "Time {}".format( elapsed ) )

            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
               'Wasserstein distance': WassersteinD.data.item( ),
               'Loss D': lossD.data.item( ),
               'Loss G': costG.data.item( ),
               'Loss D Real': lossRealD.data.item( ),
               'Loss D Fake': lossFakeD.data.item( )

            }

            for tag, value in info.items( ):
               self.logger.scalar_summary( tag, value, iterG + 1 )

            # (3) Log the images
            info = {
               'real_images': self.real_images( images, self.numberOfImages ),
               'generated_images': self.generate_img( z, self.numberOfImages )
            }

            for tag, images in info.items( ):
               self.logger.image_summary( tag, images, iterG + 1 )

      self.end = t.time( )
      print( 'Time of training-{}'.format( ( self.end - self.begin ) ) )

      # Save the trained parameters
      self.save_model( )

   def CalculateGradientPenalty( self, realImages, fakeImages ):
      eta = torch.FloatTensor( self.batchSize, 1, 1, 1 ).uniform_( 0, 1 )
      eta = eta.expand( self.batchSize, realImages.size( 1 ), realImages.size( 2 ), realImages.size( 3 ) )
      if( self.cudaEnable ):
         eta = eta.cuda( self.cudaIndex )

      interpolated = eta * realImages + ( ( 1 - eta ) * fakeImages )

      if( self.cudaEnable ):
         interpolated = interpolated.cuda( self.cudaIndex )

      # define it to calculate gradient
      if( self.cudaEnable ):
         interpolated = Variable( interpolated, requires_grad = True ).cuda( self.cudaIndex )
      else:
         interpolated = Variable( interpolated, requires_grad = True )

      # calculate probability of interpolated examples
      prob_interpolated = self.D( interpolated )

      # calculate gradients of probabilities with respect to examples
      gradients = autograd.grad( outputs = prob_interpolated, inputs = interpolated,
                                 grad_outputs = torch.ones( prob_interpolated.size( ) ).cuda( self.cudaIndex ) if self.cudaEnable else 
                                                torch.ones( prob_interpolated.size( ) ),
                                                create_graph = True, 
                                                retain_graph = True )[ 0 ]

      gradPenalty = ( ( gradients.norm( 2, dim = 1 ) - 1 ) ** 2 ).mean( ) * self.lambdaTerm
      return( gradPenalty )