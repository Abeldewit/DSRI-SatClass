from locale import normalize
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import os, sys
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, JaccardIndex
import numpy as np
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import torchvision
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.getcwd())
sys.path.insert(0, root_dir)
from src.backbones.UNet.unet import conv_block, encoder_block, decoder_block, UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.backbones.Vit.model.pretrained_segmenter import PreSegmenter
from src.utilities.dataloader import PASTIS
# from src.utilities.diceloss import DiceLoss
from segmentation_models_pytorch.losses import DiceLoss
del sys.path[0]

class LitModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        data_args: dict,
        path: str,
        num_workers=4,
        batch_size=4,
        learning_rate=None,
        # loss_function = torch.nn.CrossEntropyLoss(label_smoothing=.1),
        loss_function = None,
        save_dir = './models/',
        hparams=None,
        image_scale=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn_train = DiceLoss(
            mode='multiclass',
        ) if loss_function is None else loss_function
        self.loss_fn_val = DiceLoss(
            mode='multiclass',
        ) if loss_function is None else loss_function
        self.user_params = hparams
        self.create_metrics()

        self.standard_args = {
            'data_files': 'DATA_S2',
            'label_files':'ANNOTATIONS',
        }
        self.standard_args.update({'path_to_pastis': path})

        self.data_args = data_args

        self.best_vloss = float('inf')
        self.save_dir = save_dir

        if image_scale is not None:
            self.image_transform = torchvision.transforms.Resize(image_scale)
        else:
            self.image_transform = torchvision.transforms.Resize((128, 128))
        
        self.model = model


    def create_metrics(self):
        # Training metrics
        self._precision_train = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self._recall_train = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self._accuracy_train = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self._f1_train = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self._jaccard_train = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')

        # Validation metrics
        self._precision_val = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self._recall_val = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self._accuracy_val = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self._f1_val = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self._jaccard_val = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')

        # Test metrics
        self._precision_test = Precision(num_classes=20, average='macro', mdmc_average='samplewise')
        self._recall_test = Recall(num_classes=20, average='macro', mdmc_average='samplewise')
        self._accuracy_test = Accuracy(num_classes=20, average='weighted', mdmc_average='samplewise')
        self._f1_test = F1Score(num_classes=20, average='macro', mdmc_average='samplewise')
        self._jaccard_test = JaccardIndex(num_classes=20, average='weighted', mdmc_average='samplewise')

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

    # def on_after_backward(self):
    #     if isinstance(self.logger,TensorBoardLogger):
    #         global_step = self.global_step
    #         for name, param in self.model.named_parameters():
    #             self.logger.experiment.add_histogram(name, param, global_step)
    #             if param.requires_grad:
    #                 if param.grad is not None:
    #                     self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def train_dataloader(self):
        train_set = PASTIS(
            **self.standard_args, 
            **self.data_args, 
            subset_type='train',
            shuffle=True,
            norm=self.user_params.norm
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
            subset_type='val',
            norm=self.user_params.norm,
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

    def test_dataloader(self):
        test_set = PASTIS(
            **self.standard_args, 
            **self.data_args, 
            subset_type='test',
            norm=self.user_params.norm,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.0001,

            )

        if isinstance(self.model, PreSegmenter):
            params = self.model.parameters() if self.model.fine_tune else self.model.decoder.parameters()
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0.0001,
            )
        self.optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.user_params.monitor_mode,
            factor=0.5,
            patience=np.ceil(self.user_params.patience / 2),,
            verbose=True,
            threshold=self.user_params.min_delta,
            threshold_mode='rel',
            min_lr=0.,
            eps=1e-8,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": self.user_params.monitor,
            "strict": False,
            "name": None,
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
     
    def training_step(self, train_batch, batch_idx):
        inputs, labels, times = train_batch
        self.model.train(True)
        if isinstance(self.model, PreSegmenter):
            # Increase image size
            if self.data_args['multi_temporal']:
                og_shape = inputs.shape
                inputs = inputs.view(inputs.shape[0], inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])
                inputs = self.image_transform(inputs)
                inputs = inputs.view(*og_shape[:3], inputs.shape[-2], inputs.shape[-1])
            else:
                inputs = self.image_transform(inputs)

        outputs = self(inputs, times)
        loss = self.loss_fn_train(outputs, labels.long())
        self.log("metrics/batch/loss", loss, prog_bar=False)

        # Update metrics
        accuracy = self._accuracy_train(outputs, labels.int())
        precision = self._precision_train(outputs, labels.int())
        recall = self._recall_train(outputs, labels.int())
        f1 = self._f1_train(outputs, labels.int())
        jaccard = self._jaccard_train(outputs, labels.int())

        # Log metrics
        self.log("metrics/batch/acc", accuracy)
        self.log("metrics/batch/precision", precision)
        self.log("metrics/batch/recall", recall)
        self.log("metrics/batch/f1", f1)
        self.log("metrics/batch/jaccard", jaccard)

        return loss

    def validation_step(self, val_batch, batch_idx):
        vinputs, vlabels, vtimes = val_batch
        self.model.eval()
        if isinstance(self.model, PreSegmenter):
            # Increase image size
            if self.data_args['multi_temporal']:
                og_shape = vinputs.shape
                vinputs = vinputs.view(vinputs.shape[0], vinputs.shape[1]*vinputs.shape[2], vinputs.shape[3], vinputs.shape[4])
                vinputs = self.image_transform(vinputs)
                vinputs = vinputs.view(*og_shape[:3], vinputs.shape[-2], vinputs.shape[-1])
            else:
                vinputs = self.image_transform(vinputs)

        voutputs = self(vinputs, vtimes)
        vloss = self.loss_fn_val(voutputs, vlabels.long())
        if hasattr(self, 'optimizer'):
            self.log("metrics/val/loss", vloss, prog_bar=False, on_step=False, on_epoch=True)
        else:
            self.log("metrics/val/loss", vloss, prog_bar=True, on_step=True, on_epoch=True)

        # Update metrics
        self._accuracy_val(voutputs, vlabels.int())
        self._precision_val(voutputs, vlabels.int())
        self._recall_val(voutputs, vlabels.int())
        self._f1_val(voutputs, vlabels.int())
        self._jaccard_val(voutputs, vlabels.int())

        # Save model if validation loss is lower
        if vloss < self.best_vloss and hasattr(self, 'optimizer'):
            self.best_vloss = vloss
            self.save_model(vloss)
        
        return vloss

    def on_validation_end(self):
        pass

    def validation_epoch_end(self, outputs):
        # Log metrics
        # loss = np.array([])
        # for results_dict in outputs:
        #     loss = np.append(loss, results_dict["loss"])
        
        # # self.logger.experiment["val/loss"] = loss.mean()
        # self.log("metrics/val/loss", loss.mean(), prog_bar=True)
        self.log("metrics/val/acc", self._accuracy_val.compute())
        self.log("metrics/val/precision", self._precision_val.compute())
        self.log("metrics/val/recall", self._recall_val.compute())
        self.log("metrics/val/f1", self._f1_val.compute())
        self.log("metrics/val/jaccard", self._jaccard_val.compute())
        
        self._accuracy_val.reset()
        self._precision_val.reset()
        self._recall_val.reset()
        self._f1_val.reset()
        self._jaccard_val.reset()

    def test_step(self, test_batch, batch_idx):
        self.model.eval()
        tinputs, tlabels, ttimes = test_batch
        if isinstance(self.model, PreSegmenter):
            # Increase image size
            if self.data_args['multi_temporal']:
                og_shape = tinputs.shape
                tinputs = tinputs.view(tinputs.shape[0], tinputs.shape[1]*tinputs.shape[2], tinputs.shape[3], tinputs.shape[4])
                tinputs = self.image_transform(tinputs)
                tinputs = tinputs.view(*og_shape[:3], tinputs.shape[-2], tinputs.shape[-1])
            else:
                tinputs = self.image_transform(tinputs)
        toutputs = self(tinputs, ttimes)

        # Update metrics
        self._accuracy_test(toutputs, tlabels.int())
        self._precision_test(toutputs, tlabels.int())
        self._recall_test(toutputs, tlabels.int())
        self._f1_test(toutputs, tlabels.int())
        self._jaccard_test(toutputs, tlabels.int())

    def on_test_epoch_end(self) -> None:
        self.log("metrics/test/acc", self._accuracy_test.compute())
        self.log("metrics/test/precision", self._precision_test.compute())
        self.log("metrics/test/recall", self._recall_test.compute())
        self.log("metrics/test/f1", self._f1_test.compute())
        self.log("metrics/test/jaccard", self._jaccard_test.compute())

    def save_model(self, loss):
        model_name = str(type(self.model)).split('.')[2]
        if str(type(self.model)).split('.')[-1] == "PreSegmenter'>":
            model_name = "PViT"
        name = model_name + "_" + str(list(self.data_args.values()))
        if not os.path.exists(os.path.join(self.save_dir, model_name)):
            os.makedirs(os.path.join(self.save_dir, model_name))

        path = os.path.join(self.save_dir, model_name, name) + '.ckpt'
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                }, path)

