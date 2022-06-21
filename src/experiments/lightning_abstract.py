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
            verbose=True,
            threshold=1e-4,
            threshold_mode='rel',
            min_lr=0.,
            eps=1e-8,
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

        # Update metrics
        train_acc = self.accuracy_train(outputs, labels.int())
        train_prec = self.precision_train(outputs, labels.int())
        train_rec = self.recall_train(outputs, labels.int())
        train_f1 = self.f1_train(outputs, labels.int())
        train_jac = self.jaccard_train(outputs, labels.int())

        # Log metrics
        self.log(f'Loss/train', loss, prog_bar=True)
        self.log(
            f"Performance/train", {
                'train_acc': train_acc,
                'train_prec': train_prec,
                'train_rec': train_rec,
                'train_f1': train_f1,
                'train_jaccard': train_jac,
            }
        )

        return loss

    def on_train_epoch_end(self):
        self.accuracy_train.reset()
        self.precision_train.reset()
        self.recall_train.reset()
        self.f1_train.reset()
        self.jaccard_train.reset()

    def validation_step(self, val_batch, batch_idx):
        vinputs, vlabels, vtimes = val_batch
        voutputs = self(vinputs, vtimes)
        vloss = self.loss_fn(voutputs, vlabels.long())

        # Update metrics
        self.log(f'Loss/val', vloss, prog_bar=True)
        self.accuracy_val.update(voutputs, vlabels.int())
        self.precision_val.update(voutputs, vlabels.int())
        self.recall_val.update(voutputs, vlabels.int())
        self.f1_val.update(voutputs, vlabels.int())
        self.jaccard_val.update(voutputs, vlabels.int())

        return vloss

    def on_validation_epoch_end(self):
        self.log(
            f"Performance/val", {
                'val_acc': self.accuracy_val.compute(),
                'val_prec': self.precision_val.compute(),
                'val_rec': self.recall_val.compute(),
                'val_f1': self.f1_val.compute(),
                'val_jaccard': self.jaccard_val.compute(),
            }
        )

        self.accuracy_val.reset()
        self.precision_val.reset()
        self.recall_val.reset()
        self.f1_val.reset()
        self.jaccard_val.reset()



