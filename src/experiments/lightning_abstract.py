import torch
import pytorch_lightning as pl
import os, sys
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, JaccardIndex

root_dir = os.path.abspath(os.getcwd())
sys.path.insert(0, root_dir)
from src.backbones.UNet.unet import conv_block, encoder_block, decoder_block, UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.utilities.dataloader import create_split_dataloaders, PASTIS
del sys.path[0]

class LitModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        data_args: dict,
        path: str,
        num_workers=4,
        batch_size=4,
        learning_rate=0.05,
        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=.1),
    ):
        super().__init__()
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_function

        self.standard_args = {
            'data_files': 'DATA_S2',
            'label_files':'ANNOTATIONS',
        }
        self.standard_args.update({'path_to_pastis': path})

        self.data_args = data_args

        # Training metrics
        self.precision_train = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self.recall_train = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self.accuracy_train = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self.f1_train = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self.jaccard_train = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')

        # Validation metrics
        self.precision_val = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self.recall_val = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self.accuracy_val = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self.f1_val = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self.jaccard_val = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')
        
        self.model = model

    def forward(self, x, times=None):
        if isinstance(self.model, UTAE):
            return self.model(x, batch_positions=times)
        else:
            return self.model(x)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.log('lr', optimizer.param_groups[0]['lr'], prog_bar=True)

    def train_dataloader(self):
        train_set = PASTIS(
            **self.standard_args, 
            **self.data_args, 
            subset_type='train'
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        return train_loader

    def val_dataloader(self):
        val_set = PASTIS(
            **self.standard_args, 
            **self.data_args, 
            subset_type='val'
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "Loss/val",
            "strict": True,
            "name": None,
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def training_step(self, train_batch, batch_idx):
        inputs, labels, times = train_batch
        outputs = self(inputs, times)
        loss = self.loss_fn(outputs, labels.long())

        # Log metrics
        self.log_metrics(outputs, labels, loss, val=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels, times = val_batch
        outputs = self(inputs, times)
        loss = self.loss_fn(outputs, labels.long())

        # Log metrics
        self.log_metrics(outputs, labels, loss, val=True)

        return loss

    def log_metrics(self, outputs, labels, loss, val:bool = False):
        # Train or val metrics
        if not val:
            acc = self.accuracy_train
            prec = self.precision_train
            rec = self.recall_train
            f1 = self.f1_train
            jaccard = self.jaccard_train
            prefix='train'
        else:
            acc = self.accuracy_val
            prec = self.precision_val
            rec = self.recall_val
            f1 = self.f1_val
            jaccard = self.jaccard_val
            prefix='val'
        
        # Update metrics
        acc(outputs, labels.int())
        prec(outputs, labels.int())
        rec(outputs, labels.int())
        f1(outputs, labels.int())
        jaccard(outputs, labels.int())

        # Log metrics
        self.log(f'Loss/{prefix}', loss, prog_bar=True)
        self.log(f"Performance/{prefix}", {
            f'{prefix}_acc': acc,
            f'{prefix}_prec': prec,
            f'{prefix}_rec': rec,
            f'{prefix}_f1': f1,
            f'{prefix}_jaccard': jaccard,
        }, on_step=True, on_epoch=True)


