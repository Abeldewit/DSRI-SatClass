import torch
import pytorch_lightning as pl
import os, sys

root_dir = os.path.abspath(os.getcwd())
sys.path.insert(0, root_dir)
print(sys.path)
from src.backbones.UNet.unet import conv_block, encoder_block, decoder_block, UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.utilities.dataloader import create_split_dataloaders, PASTIS
del sys.path[0]

class LiTUNet(pl.LightningModule):
    def __init__(self, enc_channels=(3, 64, 128, 256, 512), bottleneck=1024, num_classes=20):
        super().__init__()
        self.model = UNet(num_classes=20)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            skip, x = self.encoders[i](x)
            skips.append(skip)
        
        x = self.bottleneck(x)

        for i in range(self.depth):
            x = self.decoders[i](x, skips[-i-1])

        out = self.out(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        inputs, labels, times = train_batch
        outputs = self.model(inputs) if not isinstance(self.model, UTAE) else self.model(inputs, times)
        loss = self.loss_fn(outputs, labels.long())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels, times = val_batch
        outputs = self.model(inputs) if not isinstance(self.model, UTAE) else self.model(inputs, times)
        loss = self.loss_fn(outputs, labels.long())
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.log('lr', optimizer.param_groups[0]['lr'], prog_bar=True)




if __name__ == "__main__":
    model = LiTUNet()
    standard_args = {
        'path_to_pastis':'/workspace/persistent/data/PASTIS/', 
        'data_files': 'DATA_S2', 
        'label_files':'ANNOTATIONS',
    }

    test_args = [
        {
            'rgb_only': True, 
            'multi_temporal': False,
            'fold': 1,
        },
    ]
    train, val, test = create_split_dataloaders(
        **standard_args, 
        **test_args[0], 
        shuffle=True, 
        batch_size=64,
        num_workers=6
    )

    trainer = pl.Trainer(max_epochs=1)
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=50)

    trainer.fit(model, train, val)