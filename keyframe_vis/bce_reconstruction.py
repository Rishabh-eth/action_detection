# This script helps to understand the Binary Cross Entropy Loss as implemented in https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
# The calculation shall result in: 0.7084


import torch
labels = torch.tensor([[1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 0.],
                       [1., 0., 0., 0., 0.]])

predictions = torch.tensor([[0.4092, 0.5203, 0.6224, 0.2949, 0.4273],
                            [0.3509, 0.5938, 0.6599, 0.3428, 0.7040],
                            [0.4885, 0.6341, 0.6177, 0.4519, 0.6910],
                            [0.4400, 0.3157, 0.3997, 0.3721, 0.5693]])

one = torch.ones([4,5])


mid1 = labels * torch.log(predictions)
mid2 = (one - labels) * torch.log(1 - predictions)

l = -(mid1 + mid2)

print(torch.mean(l))

