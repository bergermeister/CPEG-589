import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import utils
from time import time
from Utility.Logger import Logger

# Generative Adversarial Network (GAN)
class Network( object ):
   def __init__( self, Generator, Discriminator, OptimizerG, OptimizerD, Args ):
      print( "GAN Model Initialization" )

      # Store the Generator, Discriminator, and Channel information
      self.G          = Generator       # Set Generator
      self.D          = Discriminator   # Set Discriminator
      self.C          = Args.channels   # Set Channels
      self.optimizerG = OptimizerG      # Set Generator Optimizer
      self.optimizerD = OptimizerD      # Set Discriminator Optimizer

      # Binary cross entropy loss and optimizer
      self.loss = nn.BCELoss( )

      # Update CUDA configuration
      self.cudaIndex = 0
      self.cudaEnable = False
      self.UpdateCuda( Args.cuda )
      
      # Set Logger
      self.logger = Logger( '.logs' )
      
      # Store argument configuration
      self.numberOfImages = 10
      self.epochs = Args.epochs
      self.batchSize = Args.batch_size

   def UpdateCuda( self, cudaEnable = False ):
      if( cudaEnable == True ):
         self.cudaIndex = 0
         self.cudaEnable = True
         self.D.cuda( self.cudaIndex )
         self.G.cuda( self.cudaIndex )
         self.loss = nn.BCELoss( ).cuda( self.cudaIndex )

   def GetVariable( self, arg ):
      var = None
      if( self.cudaEnable == True ):
         var = Variable( arg ).cuda( self.cudaIndex )
      else:
         var = Variable( arg )
      return( var )
         
   def Train( self, loader ):
      self.begin = time( )
      iterG = 0             # Generator iteration
      
      for epoch in range( self.epochs ):
         self.epochStart = time( )
         
         for i, (images, _) in enumerate( loader ):
            # Check if round number of batches
            if( i == loader.dataset.__len__( ) // self.batchSize ):
               break
               
            images     = self.GetVariable( images.view( self.batchSize, -1 ) )
            realLabels = self.GetVariable( torch.ones( self.batchSize ) )
            fakeLabels = self.GetVariable( torch.zeros( self.batchSize ) )
            
            # Train discriminator
            # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # [Training discriminator = Maximizing discriminator being correct]
            outputs = self.D( images )
            lossReal = self.loss( outputs, realLabels )
            real_score = outputs

            # Compute BCE Loss using fake images
            z          = self.GetVariable( torch.randn( ( self.batchSize, 100, 1, 1 ) ).view( self.batchSize, -1 ) )
            fakeImages = self.G( z )
            outputs    = self.D( fakeImages )
            lossFake   = self.loss( outputs, fakeLabels )
            fake_score = outputs

            # Optimize discriminator
            lossD = lossReal + lossFake
            self.D.zero_grad( )
            lossD.backward( )
            self.optimizerD.step( )

            # Train generator
            # Compute loss with fake images
            z          = self.GetVariable( torch.randn( ( self.batchSize, 100, 1, 1 ) ).view( self.batchSize, -1 ) )
            fakeImages = self.G( z )
            outputs    = self.D( fakeImages )
            lossG      = self.loss( outputs, realLabels )

            # Optimize generator
            self.D.zero_grad( )
            self.G.zero_grad( )
            lossG.backward( )
            self.optimizerG.step( )
            iterG += 1

            if( ( iterG % 1000 ) == 0 ):
               # Workaround because graphic card memory can't store more than 800+ examples in memory for generating image
               # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
               # This way Inception score is more correct since there are different generated examples from every class of Inception model
               # sample_list = []
               # for i in range(10):
               #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
               #     samples = self.G(z)
               #     sample_list.append(samples.data.cpu().numpy())
               #
               # # Flattening list of lists into one list of numpy arrays
               # new_sample_list = list(chain.from_iterable(sample_list))
               # print("Calculating Inception Score over 8k generated images")
               # # Feeding list of numpy arrays
               # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
               #                                       resize=True, splits=10)
               print( 'Epoch-{}'.format( epoch + 1 ) )
               self.save_model( )

               if not os.path.exists('training_result_images/'):
                  os.makedirs('training_result_images/')

               # Denormalize images and save them in grid 8x8
               z       = self.GetVariable( torch.randn( self.batchSize, 100 ) )               
               samples = self.G( z )
               samples = samples.mul( 0.5 ).add( 0.5 )
               samples = samples.data.cpu( )[ :64 ]
               grid = utils.make_grid( samples )
               utils.save_image( grid, 'training_result_images/img_generatori_iter_{}.png'.format( str( iterG ).zfill( 3 ) ) )

               elapsed = time( ) - self.begin
               print( "Generator iter: {}".format( iterG ) )
               print( "Time {}".format( elapsed ) )

            if( ( ( i + 1 ) % 100 ) == 0 ):
               print( "Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                      ( ( epoch + 1 ), ( i + 1 ), loader.dataset.__len__( ) // self.batchSize, lossD.data.item( ), lossG.data.item( ) ) )
               
               z = self.GetVariable( torch.randn( ( self.batchSize, 100, 1, 1 ) ).view( self.batchSize, -1 ) )

               # TensorBoard logging
               # Log the scalar values
               info = {
                  'lossD': lossD.data.item( ),
                  'lossG': lossG.data.item( ) 
               }

               for tag, value in info.items( ):
                  self.logger.scalar_summary( tag, value, iterG )

               # Log values and gradients of the parameters
               for tag, value in self.D.named_parameters( ):
                  tag = tag.replace( '.', '/' )
                  self.logger.histo_summary( tag, ToNP(value), iterG)
                  self.logger.histo_summary( tag + '/grad', ToNP(value.grad), iterG)

               # Log the images while training
               info = {
                  'real_images': self.real_images(images, self.numberOfImages),
                  'generated_images': self.generate_img(z, self.numberOfImages)
               }

               for tag, images in info.items():
                  self.logger.image_summary(tag, images, iterG)


      self.end = time( )
      print('Time of training-{}'.format((self.end - self.begin)))
      #self.file.close()

      # Save the trained parameters
      self.save_model()

   def evaluate( self, test_loader, D_model_path, G_model_path ):
      self.load_model(D_model_path, G_model_path)
      z = self.GetVariable( torch.randn( self.batchSize, 100, 1, 1 ) )
      samples = self.G( z )
      samples = samples.mul( 0.5 ).add( 0.5 )
      samples = samples.data.cpu( )
      grid = utils.make_grid( samples )
      print( "Grid of 8x8 images saved to 'dgan_model_image.png'." )
      utils.save_image( grid, 'dgan_model_image.png' )

   def real_images( self, images, numberOfImages ):
      if( self.C == 3 ):
         return ToNP( images.view( -1, self.C, 32, 32 )[ :self.numberOfImages ] )
      else:
         return ToNP( images.view( -1, 32, 32 )[ :self.numberOfImages ] )

   def generate_img( self, z, numberOfImages ):
      samples = self.G( z ).data.cpu( ).numpy( )[ :numberOfImages ]
      generated_images = [ ]
      for sample in samples:
         if self.C == 3:
            generated_images.append( sample.reshape( self.C, 32, 32 ) )
         else:
            generated_images.append( sample.reshape( 32, 32 ) )
      return( generated_images )

   def save_model( self ):
      torch.save( self.G.state_dict( ), './generator.pkl' )
      torch.save( self.D.state_dict( ), './discriminator.pkl' )
      print( 'Models save to ./generator.pkl & ./discriminator.pkl' )

   def load_model( self, D_model_filename, G_model_filename ):
      D_model_path = os.path.join( os.getcwd( ), D_model_filename )
      G_model_path = os.path.join( os.getcwd( ), G_model_filename )
      self.D.load_state_dict( torch.load( D_model_path ) )
      self.G.load_state_dict( torch.load( G_model_path ) )
      print( 'Generator model loaded from {}.'.format( G_model_path ) )
      print( 'Discriminator model loaded from {}-'.format( D_model_path ) )
            
   def generate_latent_walk( self, number ):
      if not os.path.exists( 'interpolated_images/' ):
         os.makedirs( 'interpolated_images/' )

      number_int = 10
      # interpolate between twe noise(z1, z2).
      z_intp = torch.FloatTensor( 1, 100, 1, 1 )
      z1 = torch.randn( 1, 100, 1, 1 )
      z2 = torch.randn( 1, 100, 1, 1 )
      if( self.cudaEnable ):
         z_intp = z_intp.cuda( )
         z1 = z1.cuda( )
         z2 = z2.cuda( )

      z_intp = Variable( z_intp )
      images = [ ]
      alpha = 1.0 / float( number_int + 1 )
      print( alpha )
      for i in range( 1, number_int + 1 ):
         z_intp.data = ( z1 * alpha ) + ( z2 * ( 1.0 - alpha ) )
         alpha += alpha
         fake_im = self.G( z_intp )
         fake_im = fake_im.mul( 0.5 ).add( 0.5 ) #denormalize
         images.append( fake_im.view( self.C, 32, 32 ).data.cpu( ) )
      
      grid = utils.make_grid( images, nrow = number_int )
      utils.save_image( grid, 'interpolated_images/interpolated_{}.png'.format( str( number ).zfill( 3 ) ) )
      print( "Saved interpolated images." )
        
def ToNP( x ):
   return( x.data.cpu( ).numpy( ) )

def GetInfiniteBatches( DataLoader ):
   while True:
      for i, ( images, _ ) in enumerate( DataLoader ):
         yield images
