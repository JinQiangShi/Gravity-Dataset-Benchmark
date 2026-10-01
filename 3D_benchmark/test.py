import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from src.dataloader import ZarrDataloader, dataset_path
density = ZarrDataloader(dataset_path("geo_model", "density1"), batch_size=2)
for data, model in density.train_dataloader:
    print(data.shape, model.shape)
    break

from src.model.unet import UNet
unet = UNet(in_channels=3*3, out_channels=128)
output = unet(data)
print(output.shape)
