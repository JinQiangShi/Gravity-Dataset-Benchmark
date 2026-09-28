import torch
from src.model.unetpp import UNetPP

model = UNetPP(in_channels=2, out_channels=128, deep_supervision=True)
x = torch.zeros((4, 2, 128))
y1, y2, y3, y4 = model(x)
print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)
