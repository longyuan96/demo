import torch.utils.data
import torchvision.transforms as transforms
from model import LeNet
from PIL import Image

transform = transforms.Compose([transforms.Resize((32,32)),
    transforms.ToTensor(), transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # ToTensor 0-1

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('lenet.pth'))

im = Image.open('1.JPG')
im = transform(im) #[c,w,h]
im = torch.unsqueeze(im,dim=0) #[batch,c,w,h]

with torch.no_grad():
    outputs = net(im)
    pred = torch.max(outputs,dim=1)[1].data.numpy()


print(classes[int(pred)])