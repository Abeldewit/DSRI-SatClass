import torch
import sys, os
import getopt

sys.path.insert(0, os.getcwd())
from src.backbones.UNet.unet import UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.experiments.model_trainer import train_model
from src.utilities.dataloader import PASTIS
del sys.path[0]
import json
from pyifttt.webhook import send_notification
from torch.utils.data import DataLoader

all_args = sys.argv[1:]
try:
    # Gather the arguments
    opts, args = getopt.getopt(
        all_args,
        'e:p:', 
        [
            'epochs=', 
            'path=',
            'start=',
            'end=',
            'key=',
            'num_workers=',
        ])
except:
    print('Error parsing arguments')

# PARAMETERS
learning_rate = 0.01
SHUFFLE = True
MODEL_DIR = './models/'
LOG_DIR = './logs/tensorboard/'
EPOCHS = None
PATH = None
START = None
END = None
NUMWORK = None
KEY = None
for opt, arg in opts:
    if opt in ('-e', '--epochs'):
        EPOCHS = int(arg)
    elif opt in ('-p', '--path'):
        PATH = arg
    elif opt in ('--start',):
        START = max(int(arg) - 1, 0)
    elif opt in ('--end',):
        END = int(arg)
    elif opt in ('--num_workers',):
        NUMWORK = int(arg)
    elif opt in ('--key',):
        os.environ['IFTTT_KEY'] = arg
        try:
            send_notification('python_notification', data={'value1': 'Started training'})
        except Exception as e:
            print('Error sending notification')

if EPOCHS is None or PATH is None:
    print('Error parsing arguments')
    sys.exit()

# Standard arguments for every experiment
STD_ARGS = {
        'path_to_pastis': PATH, 
        'data_files': 'DATA_S2', 
        'label_files':'ANNOTATIONS',
}

# Create folders for models and logs
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# Read in the experiments
with open(os.path.join(os.getcwd(), 'src/experiments/experiments.json'), 'r') as f:
    experiments = json.load(f)

# Set the start and end if given
if START is not None and END is not None:
    experiments = dict(list(experiments.items())[START:END])
elif START is not None:
    experiments = dict(list(experiments.items())[START:])
elif END is not None:
    experiments = dict(list(experiments.items())[:END])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {
    'UNet': UNet,
    'UTAE': UTAE,
    'ViT': {
        'encoder': VisionTransformer, 
        'decoder': MaskTransformer, 
        'segmenter': Segmenter
        },
}


for exp, args in list(experiments.items()):
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

    # Create the datasets
    train_set = PASTIS(**STD_ARGS, **data_options, shuffle=True, subset_type='train')
    val_set = PASTIS(**STD_ARGS, **data_options, shuffle=True, subset_type='val')

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=SHUFFLE,
        num_workers=NUMWORK,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUMWORK,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Creating the model
    if model_name != 'ViT':  # Vit needs some special attention
        model_options = {}
        model_options.update(args['standard_arguments'])
        model_options.update(args['special_arguments'])

        model = models[model_name](**model_options)
    else:
        encoder_options = args['standard_arguments']['encoder']
        decoder_options = args['standard_arguments']['decoder']
        segmenter_options = args['standard_arguments']['segmenter']
        encoder_options.update(args['special_arguments']['encoder'])

        model = models[model_name]['segmenter'](
            encoder=models[model_name]['encoder'](**encoder_options),
            decoder=models[model_name]['decoder'](**decoder_options),
            **segmenter_options
        )

    # Move the model to the GPU
    model.to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Create the loss function
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=.1)

    # Train the model
    name = model_name + str(list(data_options.values()))

    train_model(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        n_epochs=EPOCHS,
        batch_size=batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir='./logs/tensorboard',
        save_dir='./models/',
        name=name,
    )