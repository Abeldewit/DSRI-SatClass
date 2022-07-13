from src.backbones.UNet.unet import UNet
from src.backbones.UTAE.utae import UTAE
from src.backbones.Vit.model.vit import VisionTransformer
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.segmenter import Segmenter
from src.experiments.lightning_abstract import LitModule
from src.backbones.Vit.model.pretrained_segmenter import EncoderVit, PreSegmenter

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
        'PViT': {
            'encoder': EncoderVit,
            'decoder': MaskTransformer,
            'segmenter': PreSegmenter
        }
    }
    # Creating the model
    if model_name == 'UNet' or model_name == 'UTAE':
        model_options = {}
        model_options.update(args['standard_arguments'])
        model_options.update(args['special_arguments'])

        model = models[model_name](**model_options)
    elif model_name == 'ViT': # Vit needs some special attention
        encoder_options = args['standard_arguments']['encoder']
        decoder_options = args['standard_arguments']['decoder']
        segmenter_options = args['standard_arguments']['segmenter']
        encoder_options.update(args['special_arguments']['encoder'])

        model = models[model_name]['segmenter'](
            encoder=models[model_name]['encoder'](**encoder_options),
            decoder=models[model_name]['decoder'](**decoder_options),
            **segmenter_options
        )
    elif model_name == 'PViT':
        encoder_options = args['standard_arguments']['encoder']
        decoder_options = args['standard_arguments']['decoder']
        for key in args['special_arguments'].keys():
            args['standard_arguments'][key].update(args['special_arguments'][key])

        image_size = args['standard_arguments']['segmenter']['image_size']
        patch_size = args['standard_arguments']['decoder']['patch_size']
        num_patches = (image_size[0] // patch_size) ** 2
        args['standard_arguments']['encoder'].update({'embedding_dim': num_patches})

        model = models[model_name]['segmenter'](
            encoder=models[model_name]['encoder'](**args['standard_arguments']['encoder']),
            decoder=models[model_name]['decoder'](**args['standard_arguments']['decoder']),
            **args['standard_arguments']['segmenter']
        )
    return model
