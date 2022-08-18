
__all__ = ["REDCNN", "UNET", "CycleGAN" ,"WGANVGG"]

from . import REDCNN
from . import UNET
from . import CycleGAN
from . import GAN

redcnn_tmi = REDCNN.REDCNN_TMI
redcnn_bn = REDCNN.REDCNN_BN
redcnn_sa = REDCNN.REDCNN_SA
unet_mp = UNET.UNET_MP
unet_tmi = UNET.UNET_TMI
unet_mia = UNET.UNET_MIA
wganvgg = GAN.WGANVGG
cycle = CycleGAN.CycleGAN