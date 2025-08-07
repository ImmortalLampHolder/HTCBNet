from torchsummary import summary
from nets.network import HTCBNet

net = HTCBNet()
net.to('cuda')
summary(net,(1,400))


