from PIL import Image
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
import math

from augment import *

def lprint(ms, log=None):
    '''Print message on console and in a log file'''
    print(ms)
    if log:
        log.write(ms+'\n')
        log.flush()

def resize_im(wo, ho, imsize=None, dfactor=1, value_to_scale=max, enforce=False):
    wt, ht = wo, ho

    # Resize only if the image is too big
    resize = imsize and value_to_scale(wo, ho) > imsize and imsize > 0
    if resize or enforce:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])
    scale = [wo / wt, ho / ht]
    return wt, ht, scale

def read_im(im_path, imsize=None, dfactor=1):
    im = Image.open(im_path)
    im = im.convert('RGB')

    # Resize
    wo, ho = im.width, im.height
    wt, ht, scale = resize_im(wo, ho, imsize=imsize, dfactor=dfactor)
    im = im.resize((wt, ht), Image.BICUBIC)
    return im, scale

def read_im_gray(im_path, imsize=None):
    im, scale = read_im(im_path, imsize)
    return im.convert('L'), scale

def load_gray_scale_tensor(im_path, device, imsize=None, dfactor=1):
    
    im_rgb, scale = read_im(im_path, imsize, dfactor=dfactor)
    gray = np.array(im_rgb.convert('L'))
    gray = transforms.functional.to_tensor(gray).unsqueeze(0).to(device)
    return gray, scale

def sliding_window(image, window_ratio, resize):
    h, w = image.shape
    height = int(h*2/3)
    width = int(height*window_ratio)
    windows = []

    right_edge = width
    i = 0
    while i + width <= w:
        windows += [image[:height, i:i+width], image[1-height:, i:i+width]]
        
    windows = np.stack(windows).transpose((1, 2, 0))
    windows = cv2.resize(windows, resize)
    
    if len(windows.shape) == 2:
        windows = windows[None, :, :]
    else:
        windows = windows.transpose((2, 0, 1))

    return windows

def fragment_reshape(image, num, nh, nw):
    img_frags = []
    h, w = image.shape
    bh = h // num
    bw = w // num
    for r in range(num):
        for c in range(num):
            img_frags.append(image[r*bh:(r+1)*bh, c*bw:(c+1)*bw])

    img_frags = np.stack(img_frags).transpose((1, 2, 0))
    img_frags = cv2.resize(img_frags, (nw, nh))

    if len(img_frags.shape) == 2:
        img_frags = img_frags[None, :, :]
    else:
        img_frags = img_frags.transpose((2, 0, 1))
    return img_frags

def load_gray_scale_tensor_cv_augment(im_path, device, imsize=None, value_to_scale=min, dfactor=1, pad2sqr=False, split=1):
    '''Image loading function applicable for LoFTR & Aspanformer. '''
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    #im = clahe.apply(im)
    #im = rotate(im)
    #im = slit(im, split)
    ho, wo = im.shape
    ratio = wo/ho

    wt, ht, scale = resize_im(
        wo, ho, imsize=imsize, dfactor=dfactor,
        value_to_scale=value_to_scale,
        enforce=pad2sqr
    )
    im = cv2.resize(im, (wt, ht))

    #apply clache to all
    #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    
    #im = scale_brightness(im)
    #im = blur(im)

    mask = None
    if pad2sqr and (wt != ht):
        # Padding to square image
        im, mask = pad_bottom_right(im, max(wt, ht), ret_mask=True)
        mask = torch.from_numpy(mask).to(device)
    #im = transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    return im, ratio, mask

def load_gray_scale_tensor_cv(im_path, device, imsize=None, value_to_scale=min, dfactor=1, pad2sqr=False):
    '''Image loading function applicable for LoFTR & Aspanformer. '''

    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))
    #im = clahe.apply(im)

    ho, wo = im.shape
    ratio = wo/ho

    wt, ht, scale = resize_im(
        wo, ho, imsize=imsize, dfactor=dfactor,
        value_to_scale=value_to_scale,
        enforce=pad2sqr
    )
    im = cv2.resize(im, (wt, ht))
    
    #im = scale_brightness(im) #put this here for experiment with blur + mean0,std 1

    mask = None
    if pad2sqr and (wt != ht):
        # Padding to square image
        im, mask = pad_bottom_right(im, max(wt, ht), ret_mask=True)
        mask = torch.from_numpy(mask).to(device)
    im = transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    return im, ratio, mask

def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask

def load_im_tensor(im_path, device, imsize=None, normalize=True,
                   with_gray=False, raw_gray=False, dfactor=1):
    im_rgb, scale = read_im(im_path, imsize, dfactor=dfactor)

    # RGB  
    im = transforms.functional.to_tensor(im_rgb)
    if normalize:
        im = transforms.functional.normalize(
            im , mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    im = im.unsqueeze(0).to(device)
    
    if with_gray:
        # Grey
        gray = np.array(im_rgb.convert('L'))
        if not raw_gray:
            gray = transforms.functional.to_tensor(gray).unsqueeze(0).to(device)
        return im, gray, scale
    return im, scale

def load_pair_tensor(im_q, r_path, device, imsize=None, value_to_scale=min, dfactor=1, split=1, pad2sqr=False, aug_q=False, aug_r=False):
    im_r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
    ho_r, wo_r = im_r.shape
    
    wt_r, ht_r, scale_r = resize_im(
        wo_r, ho_r, imsize=imsize, dfactor=dfactor,
        value_to_scale=value_to_scale,
        enforce=pad2sqr
    )

    im_r = fragment_reshape(im_r, split, ht_r, wt_r)
    #START apply augments
    #im_r = hist_match(im_r, im_q)

    if aug_q:
        pass

    if aug_r:
        im_r = blur(im_r)

    #END apply augments
    pad_size = max(wt_r, ht_r)
    mask_r = None
    if pad2sqr and (wt_r != ht_r):
        # Padding to square image
        im_r, mask_r = pad_bottom_right(im_r, max(wt_r, ht_r), ret_mask=True)
        mask_r = torch.from_numpy(mask_r).to(device)
    elif mask_r == None:
        mask_r = torch.ones(ht_r, wt_r)[None, :, :]
        mask_r = mask_r.expand((split**2, pad_size, pad_size)).to(device)
    
    im_q = transforms.functional.to_tensor(im_q).unsqueeze(0)
    im_q = im_q.expand((split**2, 1, pad_size, pad_size)).to(device) 
    

    
    im_r = im_r.transpose(((1, 2, 0)))
    im_r = transforms.functional.to_tensor(im_r).unsqueeze(1)
    im_r = im_r.float().to(device) #change this
    torch.cuda.empty_cache()
    return im_q, im_r, scale_r, mask_r