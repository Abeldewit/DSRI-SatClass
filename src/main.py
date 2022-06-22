import torch
import pytorch_lightning as pl
import sys, os
import getopt
from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
from pyifttt.webhook import send_notification

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger


sys.path.insert(0, os.getcwd())
from src.backbones.UNet.unet import UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.experiments.lightning_abstract import LitModule
del sys.path[0]


def read_experiments():
    # Read in the experiments
    with open(os.path.join(os.getcwd(), 'src/experiments/experiments.json'), 'r') as f:
        experiments = json.load(f)
    return experiments

def experiment_generator(begin=None, end=None):
    experiments = read_experiments()
    for exp, args in list(experiments.items())[begin:end]:
        print(f'\n\n** Running: {exp} **')
        print('-'*27)
        # Get the arguments
        model_name = args['model']
        batch_size = args['batch_size']
        print(f'Model: {model_name}')

        data_options = {
            'rgb_only': args['rgb_only'],
            'multi_temporal': args['multi_temporal'],
            'fold': args['fold'],
        }

        print('Options:\n  {}'.format('\n  '.join(['{}: {}'.format(k, v) for k, v in data_options.items()])))
        print('-'*27)
        yield exp, args, data_options

def create_model(model_name, args):
    # Model setup
    models = {
        'UNet': UNet,
        'UTAE': UTAE,
        'ViT': {
            'encoder': VisionTransformer, 
            'decoder': MaskTransformer, 
            'segmenter': Segmenter
            },
    }
    # Creating the model
    if model_name != 'ViT':
        model_options = {}
        model_options.update(args['standard_arguments'])
        model_options.update(args['special_arguments'])

        model = models[model_name](**model_options)
    else: # Vit needs some special attention
        encoder_options = args['standard_arguments']['encoder']
        decoder_options = args['standard_arguments']['decoder']
        segmenter_options = args['standard_arguments']['segmenter']
        encoder_options.update(args['special_arguments']['encoder'])

        model = models[model_name]['segmenter'](
            encoder=models[model_name]['encoder'](**encoder_options),
            decoder=models[model_name]['decoder'](**decoder_options),
            **segmenter_options
        )
    return model

def create_trainer(hparams, exp):
    early_stopping = EarlyStopping(
        monitor="metrics/val/loss", 
        mode="min", 
        patience=hparams.patience,
        check_on_train_epoch_end=False,
        verbose=True,
        strict=False,
    )
    if hparams.logger == 'neptune':
        logger = NeptuneLogger(
            project="abeldewit/sat-class",
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmVlMDg0Ny0yZDI4LTQxYTUtYjU4MC02MGQ0MGIxYWM2NzEifQ==",
            log_model_checkpoints=False,
            name=exp,
        )
        logger.log_hyperparams(
            {
                'max_epochs': hparams.epochs,
                'early_stopping_patience': hparams.patience,
                'learning_rate': hparams.learning_rate,
            }
        )
    elif hparams.logger == 'tensorboard':
        logger = TensorBoardLogger('./logs', prefix=exp)
    else:
        logger = None

    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        callbacks=[early_stopping],
        logger=logger,
        fast_dev_run=hparams.fast_dev,
    )
    return trainer


def main(hparams):
    experiment_iter = experiment_generator(hparams.begin, hparams.end)

    for exp, args, data_args in experiment_iter:
        # Create the model
        model = create_model(args['model'], args)

        batch_size = hparams.batch_size if hparams.batch_size else args['batch_size']
        
        # Create lightning module
        lightning_module = LitModule(
            model=model,
            data_args=data_args,
            path=hparams.path,
            batch_size=batch_size,
            num_workers=hparams.num_workers,
            learning_rate=hparams.learning_rate,
        )

        # Create the trainer
        trainer = create_trainer(hparams, exp)

        # Run the experiment
        trainer.fit(lightning_module)

        # Test the model
        trainer.test(lightning_module)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('--accelerator', type=str, default='cpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--path', type=str, default='/workspace/persistent/data/PASTIS')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--fast_dev', default=False)
    parser.add_argument('--logger', type=str, default='neptune')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
