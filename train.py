import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import LeNet
import torch.nn as nn
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # ToTensor 0-1
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=36,
    shuffle=True,
    num_workers=0)  # num_works window 只能为0
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=10000, shuffle=False, num_workers=0)

test_data_iter = iter(testloader)  # 迭代器
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # print labels
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_image))

net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimzer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data

        optimzer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimzer.step()

        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_image)  # [batch,10]
                predict_y = torch.max(outputs, dim=1)[1]  # 只需要indxe
                ac = (predict_y == test_label).sum().item() / \
                    test_label.size(0)
                print('[%d,%5d] train_loss:%.3f test_ac:%.3f' %
                      (epoch + 1, step + 1, running_loss / 500, ac))
                running_loss = 0
print('Finshed')
save_path = './lenet.pth'
torch.save(net.state_dict(), save_path)
