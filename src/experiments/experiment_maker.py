import json
import itertools
import os

experiment_options = {
    'model': (
        'UNet',
        'UTAE', 
        # 'ViT',
        'PViT',
    ),
    'rgb_only': ('rgb', 'spec'), 
    'multi_temporal': ('temp', 'no_temp'),
    'remove_cloud': ('', 'no_cloud'), 
    'fold': (1,),
}

data_options = {
    'rgb_temp': {'rgb_only': True, 'multi_temporal': True, 'remove_clouds': False},
    'rgb_no_temp': {'rgb_only': True, 'multi_temporal': False, 'remove_clouds': False},
    'spec_temp': {'rgb_only': False, 'multi_temporal': True, 'remove_clouds': False},
    'spec_no_temp': {'rgb_only': False, 'multi_temporal': False, 'remove_clouds': False},
    'rgb_temp_no_cloud': {'rgb_only': True, 'multi_temporal': True, 'remove_clouds': True},
    'rgb_no_temp_no_cloud': {'rgb_only': True, 'multi_temporal': False, 'remove_clouds': True},
    'spec_temp_no_cloud': {'rgb_only': False, 'multi_temporal': True, 'remove_clouds': True},
    'spec_no_temp_no_cloud': {'rgb_only': False, 'multi_temporal': False, 'remove_clouds': True},
}

batch_size_options = {
    'UNet': 128,
    'UTAE': 4,
    # 'ViT': 128,
    'PViT': 64,
}

standard_model_options = {
    'UNet': {
        'num_classes': 20
    },
    'UTAE': {
        'encoder_widths':[64, 64, 64, 128],
        'decoder_widths':[32, 32, 64, 128],
        'out_conv':[32, 20],
        'str_conv_k':4,
        'str_conv_s':2,
        'str_conv_p':1,
        'agg_mode':"att_group",
        'encoder_norm':"group",
        'n_head':16,
        'd_model':256,
        'd_k':4,
        'encoder':False,
        'return_maps':False,
        'pad_value':0,
        'padding_mode':"reflect",
    },
    # 'ViT': {
    #     'encoder': {
    #         'image_size': (128,128),
    #         'patch_size': 16,
    #         'n_layers': 4,
    #         'd_model': 768,
    #         'd_ff': 256,
    #         'n_heads': 4,
    #         'n_cls': 20
    #     },
    #     'decoder': {
    #         'n_cls': 20,
    #         'patch_size': 16,
    #         'd_encoder': 768,
    #         'n_layers': 8,
    #         'd_model': 1024,
    #         'd_ff': 512,
    #         'drop_path_rate': 0.,
    #         'dropout': 0.1,
    #         'n_heads': 12,
    #     },
    #     'segmenter': {'n_cls': 20}
    # },
    'PViT': {
        'encoder': {
            'embedding_dim': 64,
            'model_dim': 1024,
            'name': 'L_16_imagenet1k',
        },
        'decoder': {
            'n_cls': 20,
            'patch_size': 16,
            'd_encoder': 1024,
            'n_layers': 8,
            'd_model': 1024,
            'd_ff': 512,
            'drop_path_rate': 0.,
            'dropout': 0.1,
            'n_heads': 8,
        },
        'segmenter': {
            'n_cls': 20,
            'image_size': (128,128),
            'fine_tune': True,
        }
    }
}

special_model_options = {
    'UNet': {
        'rgb_no_temp': {
            'enc_channels': (3, 64, 128,), 
            'bottleneck': 512, 
            'num_classes': 20
        },
        'spec_no_temp': {
            'enc_channels': (10, 64, 128),
            'bottleneck': 512,
            'num_classes': 20,
        }
    },
    'UTAE': {
        'rgb_temp': {'input_dim': 3},
        'spec_temp': {'input_dim': 10},
    },
    # 'ViT': {
    #     'rgb_no_temp': {'encoder': {'channels': 3}},
    #     'spec_no_temp': {'encoder': {'channels': 10}},
    # },
    'PViT': {
        'rgb_no_temp': {'encoder': {'pretrained': True}},
        'spec_no_temp': {'encoder': {'pretrained': False}},
    }
}





# # Get all possible combinations of the options
combinations = list(itertools.product(*experiment_options.values()))
# Combine rgb and temp options
combinations = [(c[0], '_'.join((c[1], c[2])), *c[3:]) for c in combinations]
combinations = [(c[0], '_'.join((c[1], c[2])), *c[3:]) if c[2]!='' else (c[0], c[1], *c[3:]) for c in combinations]

# Filter all possible options
combinations = [c for c in combinations if c[1].replace('_no_cloud', '') in special_model_options[c[0]].keys()]

# # Add standard model options
combinations = [(*c, standard_model_options[c[0]]) for c in combinations]

# # Add special model options
combinations = [(*c, special_model_options[c[0]][c[1].replace('_no_cloud', '')]) for c in combinations]

# Set data options
combinations = [(c[0], *data_options[c[1]].values(), *c[2:]) for c in combinations]

combinations = [(*c[:4], batch_size_options[c[0]], *c[4:]) for c in combinations]

# # All arguments
keys = [
    'model',
    'rgb_only',
    'multi_temporal',
    'remove_clouds',
    'batch_size',
    'fold',
    'standard_arguments',
    'special_arguments',
]

# # Create a dict with all the experiment arguments
all_options = {
    f'Experiment {i+1}': {keys[n]: c[n] for n in range(len(keys))} for i, c in enumerate(combinations)
}

# Write the experiments to a json file
with open(os.path.join(os.getcwd(), 'src/experiments/experiments.json'), 'w') as f:
    json.dump(all_options, f, indent=4)