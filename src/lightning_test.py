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
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, JaccardIndex

class LiTUNet(pl.LightningModule):
    def __init__(
        self,
        num_workers=4,
        batch_size=4,
        learning_rate=0.005,
    ):
        super().__init__()
        self.num_workers = num_workers
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
                'rgb_only': False,
                'multi_temporal': False,
                'fold': 1,
            },
        ]
        
        # Training metrics
        self.precision = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self.recall = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self.accuracy = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self.f1 = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self.jaccard = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')

        # Validation metrics
        self.precision_val = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self.recall_val = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self.accuracy_val = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self.f1_val = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self.jaccard_val = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')
        
        self.model = UNet(
            enc_channels=(10, 64, 128, 256, 512),
            num_classes=20
        )

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            # "interval": "epoch",
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return {
            'optimizer': optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels, times = train_batch
        outputs = self.model(inputs) if not isinstance(self.model, UTAE) else self.model(inputs, times)
        loss = self.loss_fn(outputs, labels.long())
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.accuracy(outputs, labels)
        self.log('train_acc', self.accuracy, on_step=True, on_epoch=False)
        self.precision(outputs, labels)
        self.log('train_precision', self.precision, on_step=True, on_epoch=False)
        self.recall(outputs, labels)
        self.log('train_recall', self.recall, on_step=True, on_epoch=False)
        self.f1(outputs, labels)
        self.log('train_f1', self.f1, on_step=True, on_epoch=False)
        self.jaccard(outputs, labels)
        self.log('train_jaccard', self.jaccard, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels, times = val_batch
        outputs = self.model(inputs) if not isinstance(self.model, UTAE) else self.model(inputs, times)
        loss = self.loss_fn(outputs, labels.long())

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.accuracy_val(outputs, labels)
        self.log('val_acc', self.accuracy_val, on_step=True, on_epoch=True)
        self.precision_val(outputs, labels)
        self.log('val_precision', self.precision_val, on_step=True, on_epoch=True)
        self.recall_val(outputs, labels)
        self.log('val_recall', self.recall_val, on_step=True, on_epoch=True)
        self.f1_val(outputs, labels)
        self.log('val_f1', self.f1_val, on_step=True, on_epoch=True)
        self.jaccard_val(outputs, labels)

        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.log('lr', optimizer.param_groups[0]['lr'], prog_bar=True)




if __name__ == "__main__":
    model = LiTUNet(
        batch_size=128,
        learning_rate=0.05,
        num_workers=4,
    )

    trainer = None
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1, 
            max_epochs=200,
            auto_select_gpus=True,
        )
    else:
        trainer = pl.Trainer(max_epochs=1)

    trainer.fit(model)