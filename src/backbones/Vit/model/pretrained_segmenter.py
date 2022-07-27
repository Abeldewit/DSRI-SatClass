import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_pretrained_vit as ptv
from pytorch_pretrained_vit.model import PositionalEmbedding1D
from src.backbones.Vit.model.decoder import MaskTransformer
from src.backbones.Vit.model.temporal_patch_embedder import TemporalPatchEmb

from src.backbones.Vit.model.utils import padding, unpadding

class EncoderVit(ptv.ViT):
    def __init__(self, embedding_dim, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.positional_embedding = PositionalEmbedding1D(embedding_dim, model_dim)
        
    
    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        # x = torch.cat((model.class_token.expand(1, -1, -1), x), dim=1)
        x = self.positional_embedding(x)
        x = self.transformer(x)
        return x


class PreSegmenter(nn.Module):
    def __init__(
        self, 
        decoder: MaskTransformer, 
        n_cls, 
        image_size, 
        fine_tune=True, 
        encoder=None, 
        output_image_size=None,
        multi_temporal=False,
        channels=3,
    ):
        super().__init__()
        
        embedding_size = (image_size[0] // decoder.patch_size) ** 2
        self.image_size = image_size
        self.output_size = output_image_size
        self.multi_temporal = multi_temporal
        

        self.encoder = EncoderVit(
            embedding_dim=embedding_size,
            model_dim=decoder.d_encoder,
            name='L_16_imagenet1k', 
            pretrained=True,
        ) if encoder is None else encoder

        # if multi_temporal:
        #     embedder = TemporalPatchEmb(
        #         image_size,
        #         decoder.patch_size,
        #         decoder.d_encoder,
        #         channels,
        #     )
        #     self.encoder.patch_embedding = embedder

        self.fine_tune = fine_tune
        if not self.fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.decoder = decoder
        self.n_cls = n_cls

    def __repr__(self):
        return "PreSegmenter"

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(-2), im.size(-1)
        im = padding(im, self.decoder.patch_size)
        H, W = im.size(-2), im.size(-1)

        b, t, c, h, w = im.shape
        x = self.encoder(im) if not self.multi_temporal else self.encoder(im.view(b*t, c, h, w))
        rough_masks = self.decoder(x, (H, W))
        x = x.view(b, t, c, h, w)
        x = torch.mean(x, dim=1)

        if self.output_size is None:
            masks = F.interpolate(rough_masks, size=(H, W), mode="bilinear", align_corners=True)
            masks = unpadding(masks, (H_ori, W_ori))
        else:
            masks = F.interpolate(rough_masks, size=(self.output_size[0], self.output_size[1]), mode="bicubic", align_corners=True)
            
        # masks = F.interpolate(rough_masks, size=(H, W), mode="bilinear", align_corners=True)
        # # masks = F.interpolate(masks, size=(H, W), mode="nearest-exact")

        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    