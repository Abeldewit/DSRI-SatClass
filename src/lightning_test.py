import torch
import pytorch_lightning as pl
import os, sys

root_dir = os.path.abspath(os.getcwd())
sys.path.insert(0, root_dir)
from src.backbones.UNet.unet import conv_block, encoder_block, decoder_block, UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.utilities.dataloader import create_split_dataloaders, PASTIS
del sys.path[0]

from torch.utils.data import DataLoader

class LiTUNet(pl.LightningModule):
    def __init__(
        self, 
        batch_size = 4,
        learning_rate = 0.01,
    ):
        super().__init__()
        self.model = UNet(num_classes=20)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.1)

        self.standard_args = {
            'path_to_pastis':'/workspace/persistent/data/PASTIS/', 
            'data_files': 'DATA_S2', 
            'label_files':'ANNOTATIONS',
        }

        self.test_args = [
            {
                'rgb_only': True, 
                'multi_temporal': False,
                'fold': 1,
            },
        ]

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

    def train_dataloader(self):
        train_set = PASTIS(**self.standard_args, **self.test_args[0], subset_type='train')
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
            prefetch_factor=2
        )
        return train_loader

    def val_dataloader(self):
        val_set = PASTIS(**self.standard_args, **self.test_args[0], subset_type='val')
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            prefetch_factor=2
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
    model = LiTUNet(batch_size=64)

    trainer = None
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1, 
            max_epochs=50, 
            auto_lr_find=True,
            # auto_scale_batch_size=True,
            auto_select_gpus=True,
        )
    else:
        trainer = pl.Trainer(max_epochs=1)
    # call tune to find the lr
    trainer.tune(model)

    trainer.fit(model)