import torch
import torch.nn as nn
from Model.GAN.Network import Network as GAN

# Wasserstein Generative Adversarial Network (WGAN) with Gradient Clipping (GC)
class GC( GAN ):
   def __init__( self, args ):
      print("WGAN_CP init model.")
      # Create Generator and Discriminator
      Gen = Generator( args.channels )
      Dis = Discriminator( args.channels )
      
      # Create Optimizers using using RMSprop instead of ADAM and WGAN values from paper
      d_optimizer = torch.optim.RMSprop( Dis.parameters( ), lr = 0.00005 )
      g_optimizer = torch.optim.RMSprop( Gen.parameters( ), lr = 0.00005 )
      self.weight_cliping_limit = 0.01

      # Store iteration counts
      self.generator_iters = args.generator_iters
      self.critic_iter = 5

   def train(self, train_loader):
      self.t_begin = t.time()
      #self.file = open("inception_score_graph.txt", "w")

      # Now batches are callable self.data.next()
      self.data = self.get_infinite_batches(train_loader)

      one = torch.FloatTensor([1])
      mone = one * -1
      if self.cuda:
         one = one.cuda()
         mone = mone.cuda()

      for g_iter in range(self.generator_iters):

         # Requires grad, Generator requires_grad = False
         for p in self.D.parameters():
            p.requires_grad = True

         # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
         for d_iter in range(self.critic_iter):
            self.D.zero_grad()

            # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
            for p in self.D.parameters():
               p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

            images = self.data.__next__()
            # Check for batch to have full batch_size
            if (images.size()[0] != self.batch_size):
               continue

            z = torch.rand((self.batch_size, 100, 1, 1))

            if self.cuda:
               images, z = Variable(images.cuda()), Variable(z.cuda())
            else:
               images, z = Variable(images), Variable(z)


            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            d_loss_real = self.D(images)
            d_loss_real = d_loss_real.mean(0).view(1)
            d_loss_real.backward(one)

            # Train with fake images
            if self.cuda:
               z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
            else:
               z = Variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)
            d_loss_fake = self.D(fake_images)
            d_loss_fake = d_loss_fake.mean(0).view(1)
            d_loss_fake.backward(mone)

            d_loss = d_loss_fake - d_loss_real
            Wasserstein_D = d_loss_real - d_loss_fake
            self.d_optimizer.step()


         # Generator update
         for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation

         self.G.zero_grad()

         # Train generator
         # Compute loss with fake images
         z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
         fake_images = self.G(z)
         g_loss = self.D(fake_images)
         g_loss = g_loss.mean().mean(0).view(1)
         g_loss.backward(one)
         g_cost = -g_loss
         self.g_optimizer.step()

         # Saving model and sampling images every 1000th generator iterations
         if (g_iter) % 1000 == 0:
            self.save_model()
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

            if not os.path.exists('training_result_images/'):
               os.makedirs('training_result_images/')

            # Denormalize images and save them in grid 8x8
            z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
            samples = self.G(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = utils.make_grid(samples)
            utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

            # Testing
            time = t.time() - self.t_begin
            #print("Inception score: {}".format(inception_score))
            print("Generator iter: {}".format(g_iter))
            print("Time {}".format(time))

            # Write to file inception_score, gen_iters, time
            #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
            #self.file.write(output)


            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
               'Wasserstein distance': Wasserstein_D.data[0],
               'Loss D': d_loss.data[0],
               'Loss G': g_cost.data[0],
               'Loss D Real': d_loss_real.data[0],
               'Loss D Fake': d_loss_fake.data[0]

            }

            for tag, value in info.items():
               self.logger.scalar_summary(tag, value, g_iter + 1)

            # (3) Log the images
            info = {
               'real_images': self.real_images(images, self.number_of_images),
               'generated_images': self.generate_img(z, self.number_of_images)
            }

            for tag, images in info.items():
               self.logger.image_summary( tag, images, g_iter + 1 )

      self.t_end = t.time( )
      print( 'Time of training-{}'.format( ( self.t_end - self.t_begin ) ) )
      #self.file.close()

      # Save the trained parameters
      self.save_model( )
