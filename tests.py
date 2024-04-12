import torch
from fcn_train_citysc import SimpleFCN

dummy_input = torch.randn(1, 3, 512, 512)  # Batch size 1, 3 color channels, 512x512 pixels
model = SimpleFCN(num_classes=34)
output = model(dummy_input)
print(output.shape)  # Should output torch.Size([1, num_classes, 512, 512])
