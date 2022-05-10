import numpy as np
from torch.nn import functional as F

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def get_rgb(x, batch_index=0, t_show=1):
    """Utility function to get a displayable rgb image 
    from a Sentinel-2 time series.
    """
    im = x[batch_index, t_show, [2,1,0]].cpu().numpy()
    mx = im.max(axis=(1,2))
    mi = im.min(axis=(1,2))   
    im = (im - mi[:,None,None])/(mx - mi)[:,None,None]
    im = im.swapaxes(0,2).swapaxes(0,1)
    im = np.clip(im, a_max=1, a_min=0)
    return im
