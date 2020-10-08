import torch
from Utility.Configuration import Configuration
from Model.GAN.DC.Network import Network as DCGAN
from Model.GAN.Network import Network as GAN
from Model.GAN.Generator import Generator as Generator
from Model.GAN.Discriminator import Discriminator as Discriminator
from Utility.DataLoader import get_data_loader

#from models.gan import GAN
#from models.dcgan import DCGAN_MODEL
#from models.wgan_clipping import WGAN_CP
#from models.wgan_gradient_penalty import WGAN_GP


def main(args):
   model = None
   if( args.model == 'GAN' ):
      G = Generator( args.channels )
      D = Discriminator( args.channels )
      optimizerD = torch.optim.Adam( D.parameters(), lr = 0.0002, weight_decay = 0.00001 )
      optimizerG = torch.optim.Adam( G.parameters(), lr = 0.0002, weight_decay = 0.00001 )
      model = GAN( G, D, optimizerG, optimizerD, args ) # todo: GAN(args)
   elif args.model == 'DCGAN':
      model = DCGAN( args ) # DCGAN_MODEL(args)
   elif args.model == 'WGAN-CP':
      model = None # todo: WGAN_CP(args)
   elif args.model == 'WGAN-GP':
      model = None # todo: WGAN_GP(args)
   else:
      print("Model type non-existing. Try again.")
      exit(-1)

   # Load datasets to train and test loaders
   train_loader, test_loader = get_data_loader(args)
   #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

   # Start model training
   if args.is_train == 'True':
      model.Train( train_loader )
   # start evaluating on test data
   else:
      model.evaluate(test_loader, args.load_D, args.load_G)
      #for i in range(50):
      #   model.generate_latent_walk(i)

if __name__ == '__main__':
   config = Configuration( )
   print( config.Arguments.cuda )
   main( config.Arguments )