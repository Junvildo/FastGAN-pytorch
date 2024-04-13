from models import weights_init, Discriminator, Generator
#from model_s import Generator, Discriminator
netG = Generator(ngf=64, nz=100, im_size=512)
netG.apply(weights_init)

netD = Discriminator(ndf=64, im_size=512)
netD.apply(weights_init)
freeze_list = [netD.down_from_big, netD.down_from_small, netD.down_4, netD.down_8, netD.down_16, netD.down_32, netD.down_64]
# Freeze or No Freeze
if ~1:
    for model in [netG, netD]:
        for param in model.parameters():
            param.requires_grad = True
else:
    for layer in freeze_list:
        for param in layer.parameters():
            param.requires_grad = False