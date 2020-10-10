import os
import torch
import torch.nn as nn
from torchvision import utils
from time import time
from Model.GAN.Network import ToNP
from Model.GAN.Network import Network as GAN
from Model.GAN.DC.Generator import Generator
from Model.GAN.DC.Discriminator import Discriminator

# Deep Convolutional (DC) Generative Adversarial Network (GAN)
class Network( GAN ):
   def __init__( self, args ):
      super( ).__init__( Generator( args.channels ), 
                         Discriminator( args.channels ), 
                         None, None, args )
      print("DCGAN Model Initalization")
      self.optimizerG = torch.optim.Adam( self.G.parameters( ), lr = 0.0002, betas = ( 0.5, 0.999 ) )
      self.optimizerD = torch.optim.Adam( self.D.parameters( ), lr = 0.0002, betas = ( 0.5, 0.999 ) )

   def Train( self, loader ):
      self.begin = time( )
      iterG = 0             # Generator iteration
      
      for epoch in range( self.epochs ):
         self.epochStart = time( )
         
         for i, (images, _) in enumerate( loader ):
            # Check if round number of batches
            if( i == loader.dataset.__len__( ) // self.batchSize ):
               break
               
            images     = self.GetVariable( images )
            realLabels = self.GetVariable( torch.ones( self.batchSize ) )
            fakeLabels = self.GetVariable( torch.zeros( self.batchSize ) )
            
            # Train discriminator
            # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # [Training discriminator = Maximizing discriminator being correct]
            outputs = self.D( images )
            lossReal = self.loss( outputs, realLabels )
            real_score = outputs

            # Compute BCE Loss using fake images
            z          = self.GetVariable( torch.randn( ( self.batchSize, 100, 1, 1 ) ) )
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
            z = self.GetVariable( torch.randn( ( self.batchSize, 100, 1, 1 ) ) )
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
               z       = self.GetVariable( torch.randn( 800, 100, 1, 1 ) )               
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

               z = self.GetVariable( torch.randn( self.batchSize, 100, 1, 1 ) )

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