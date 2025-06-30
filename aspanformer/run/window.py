import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
import math

from augment import *

from skimage import exposure
from skimage.exposure import match_histograms

def brightness(img):
    return round(np.mean(img), 3)

def contrast(img):
    return round(np.std(img), 3)

def brightness2(gray):
    background = np.median(gray)
    return (np.mean(gray) - background) / (background + 1e-5)

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

def slide_window(image, height, width, size, stride_ratio):
    h, w = image.shape
    windows = []
    window_loc = []

    #print(h, w, height, width)

    stride_h, stride_w = int(round(height*stride_ratio)), int(round(width*stride_ratio))

    i = 0
    while i + height <= h:
        j = 0
        while j + width <= w:
            windows += [image[i:i+height, j:j+width]]
            window_loc += [[i, height, j, width]]
            j += stride_w
            if j+width > w and j+width < w+stride_w:
                j = w - width
        i += stride_h
        if i+height > h and i+height < h+stride_h:
            i = h - height

    scale = size/max(width, height)
    resize = (int(round(width * scale)), int(round(height*scale)))
    windows = cv2.resize(np.stack(windows).transpose((1, 2, 0)), resize)

    if len(windows.shape) == 2:
        windows = windows[None, :, :]
    else:
        windows = windows.transpose((2, 0, 1))

    mask_win = None
    if height != width:
            # Padding to square image
        windows, mask_win = pad_bottom_right(windows, size, ret_mask=True)
            
    elif mask_win == None:
        mask_win = np.ones((windows.shape[0], size, size))
    return windows, mask_win, window_loc


def fine_window(image, window_ratio, frame, size):
    h, w = image.shape
    
    windows = []
    masks = []

    all_ratios = [0.5, 0.75]

    i, h_frame, j, w_frame = frame
    image_frame = image[i:i+h_frame,j:j+w_frame]
    #print('finef', frame)
    #print('frameshape', image_frame.shape, h, w)

    for r in all_ratios:
        height = int(round(h_frame * r))
        width = int(round(w_frame * r))
        
        temp_windows, mask_win, _ = slide_window(image_frame, height, width, size, 1/3)
            
        windows.append(temp_windows)
        masks.append(mask_win)

    '''
    if h > w:
        h_frame = frame[1]
        w_frame = int(w)
        
        i = frame[0]
        image_frame = image[i:i+h_frame,:]


        for r in all_ratios:
            width = int(w_frame * r)
            height = int(width/window_ratio)
            temp_windows, mask_win, _ = slide_window(image_frame, width, height, size ,2/3)
            
            windows.append(temp_windows)
            masks.append(mask_win)

    else:
        h_frame = int(h)
        w_frame = frame[1]
        i = frame[0]
        image_frame = image[:,i:i+w_frame]

        for r in all_ratios:
            height = int(h_frame * r)
            width = int(height * window_ratio)

            temp_windows, mask_win, _ = slide_window(image_frame, width, height, size, 2/3)

            windows.append(temp_windows)
            masks.append(mask_win)
    '''

    windows = np.concatenate(windows)
    masks = np.concatenate(masks)
    #print('fine', windows.shape[0])
    return windows, masks

def load_fine_windows(im_q, q_ratio, r_paths, r_indexes, curr_index, curr_loc, device, prev_overflow, batch_size=64, imsize=None, aug_q=False, aug_r=False):
    ref_batch = {}
    overflow = {}
    #print(len(r_paths))
    #print('curr loc', len(curr_loc))
    if prev_overflow is not None and prev_overflow['img'] is not None:
        if prev_overflow['img'].shape[0] > batch_size:
            ref_batch['img'] = prev_overflow['img'][:batch_size]
            ref_batch['mask'] = prev_overflow['mask'][:batch_size]
            ref_batch['id'] = [prev_overflow['index']] * batch_size
            

            overflow['img'] = prev_overflow['img'][batch_size:]
            overflow['mask'] = prev_overflow['mask'][batch_size:]
            overflow['index'] = prev_overflow['index']

        else:
            ref_batch['img'] = prev_overflow['img']
            ref_batch['mask'] = prev_overflow['mask']
            ref_batch['id'] = [prev_overflow['index']] * ref_batch['img'].shape[0]

            overflow['img'] = overflow['mask'] = None
            overflow['index'] = -1
            #i = prev_overflow['index'] - r_indexes[0] + 1
    else:
        ref_batch['img'] = ref_batch['mask'] = None
        ref_batch['id'] = []
        overflow['img'] = overflow['mask'] = None
        overflow['index'] = -1
    
    
    if prev_overflow is not None:
        if prev_overflow['index'] == -1:
            i = curr_index
        else:
            i = prev_overflow['index'] + 1
    else:
        i = curr_index

    num_refs = len(r_paths)
    #added on 5/16
    #clache = cv2.createCLAHE(clipLimit=8.0)
    #im_q = clache.apply(im_q)

    while (ref_batch['img'] is None or ref_batch['img'].shape[0] < batch_size) and i < num_refs:
        #print('fine i', i)
        #print('fine_window i', i)
        im_r = cv2.imread(r_paths[i], cv2.IMREAD_GRAYSCALE)

        #added this line on 5/16
        #im_r = match_histograms(im_r, im_q, channel_axis=None)

        im_r, mask_r = fine_window(im_r, q_ratio, curr_loc[i], imsize)

        if aug_r:
            im_r = blur(im_r)

        space = batch_size if ref_batch['img'] is None else batch_size - len(ref_batch['img'])
        if space < im_r.shape[0]:
            ref_batch['img'] = im_r[:space] if ref_batch['img'] is None else np.concatenate((ref_batch['img'], im_r[:space]))
            ref_batch['mask'] = mask_r[:space] if ref_batch['mask'] is None else np.concatenate((ref_batch['mask'], mask_r[:space]))
            ref_batch['id'] += [r_indexes[i]] * space

            overflow['img'] = im_r[space:]
            overflow['mask'] = mask_r[space:]
            overflow['index'] = r_indexes[i]
            
        else:
            ref_batch['img'] = im_r if ref_batch['img'] is None else np.concatenate((ref_batch['img'], im_r))       
            ref_batch['mask'] =  mask_r if ref_batch['mask'] is None else np.concatenate((ref_batch['mask'], mask_r))
            ref_batch['id'] += [r_indexes[i]] * im_r.shape[0]
            i += 1
    if aug_q:
        if brightness(im_q) < 75:
            clache = cv2.createCLAHE(clipLimit=16.0)
            im_q = clache.apply(im_q)
        elif contrast(im_q) < 25:
            clache = cv2.createCLAHE(clipLimit=4.0)
            im_q = clache.apply(im_q)
    
    ref_batch['img'] = ref_batch['img'].transpose(((1, 2, 0)))
    ref_batch['img'] = transforms.functional.to_tensor(ref_batch['img']).unsqueeze(1)
    ref_batch['img'] = ref_batch['img'].float().to(device) #change this

    ref_batch['mask'] = torch.from_numpy(ref_batch['mask']).to(device)
    
    im_q = transforms.functional.to_tensor(im_q).unsqueeze(0)
    q_batch = im_q.expand((ref_batch['img'].shape[0], 1, imsize, imsize)).to(device)

    torch.cuda.empty_cache()
    return q_batch, ref_batch, overflow



def coarse_window(image, window_ratio, size):
    h, w = image.shape
    windows = []
    window_loc = []
    masks = []
    #print('shape', h, w)

    full_img, full_mask, full_window_loc = slide_window(image, h, w, size, 1/3) #entire window

    windows.append(full_img)
    masks.append(full_mask)
    window_loc += full_window_loc

    #print('full', full_img.shape, full_mask.shape, full_window_loc)

    height = int(h)
    width = int(height*window_ratio)

    if width > w:
        height = int(round(height *w/width))
        width = int(w)
        
    s_windows, s_masks, s_window_loc = slide_window(image, height, width, size, 1/3) 

    windows.append(s_windows)
    masks.append(s_masks)
    window_loc += s_window_loc

    windows = np.concatenate(windows)
    masks = np.concatenate(masks)

    return windows, masks, window_loc


def load_windows(im_q, q_ratio, r_paths, r_indexes, device, prev_overflow, batch_size=64, imsize=None, aug_q=False, aug_r=False):
    ref_batch = {}
    overflow = {}
    ref_batch['img'] = None if prev_overflow is None else prev_overflow['img']
    ref_batch['mask'] = None if prev_overflow is None else prev_overflow['mask']
    ref_batch['id'] = [] if ref_batch['img'] is None else [prev_overflow['index']] * ref_batch['img'].shape[0]

    batch_info = {}

    ref_batch['loc'] = [] if ref_batch['img'] is None else prev_overflow['loc']
    overflow['img'] = overflow['mask'] = None
    overflow['loc'] = []
    overflow['path'] = []
    overflow['index'] = -1

    i = 0
    num_refs = len(r_paths)

    #clache = cv2.createCLAHE(clipLimit=8.0)
    #im_q = clache.apply(im_q)

    while (ref_batch['img'] is None or ref_batch['img'].shape[0] <= batch_size) and i < num_refs and overflow['img'] is None:
        im_r = cv2.imread(r_paths[i], cv2.IMREAD_GRAYSCALE)

        #im_r = match_histograms(im_r, im_q, channel_axis=None)

        im_r, mask_r, window_loc = coarse_window(im_r, q_ratio, imsize)

        batch_info[r_indexes[i]] = {
            'path' : r_paths[i],
            'loc' : window_loc,
            'match' : 0,
            'conf': 0
        }

        if aug_r:
            im_r = blur(im_r)

        space = batch_size if ref_batch['img'] is None else batch_size - len(ref_batch['img'])
        if space < im_r.shape[0]:
            ref_batch['img'] = im_r[:space] if ref_batch['img'] is None else np.concatenate((ref_batch['img'], im_r[:space]))
            ref_batch['mask'] = mask_r[:space] if ref_batch['mask'] is None else np.concatenate((ref_batch['mask'], mask_r[:space]))
            ref_batch['id'] += [r_indexes[i]] * space
            ref_batch['loc'] += window_loc[:space]

            overflow['img'] = im_r[space:]
            overflow['mask'] = mask_r[space:]
            overflow['loc'] += window_loc[space:]
            overflow['path'] = [r_paths[i]]
            overflow['index'] = r_indexes[i]
        else:
            ref_batch['img'] = im_r if ref_batch['img'] is None else np.concatenate((ref_batch['img'], im_r))       
            ref_batch['mask'] =  mask_r if ref_batch['mask'] is None else np.concatenate((ref_batch['mask'], mask_r))
            ref_batch['id'] += [r_indexes[i]] * im_r.shape[0]
            ref_batch['loc'] += window_loc
            i += 1
    #print(ref_batch['img'].shape, space)
    #print('oerflow', overflow['img'].shape)


    if aug_q:
        if brightness(im_q) < 75:
            clache = cv2.createCLAHE(clipLimit=16.0)
            im_q = clache.apply(im_q)
        elif contrast(im_q) < 25:
            clache = cv2.createCLAHE(clipLimit=4.0)
            im_q = clache.apply(im_q)
            
    
    ref_batch['img'] = ref_batch['img'].transpose(((1, 2, 0)))
    ref_batch['img'] = transforms.functional.to_tensor(ref_batch['img']).unsqueeze(1)
    ref_batch['img'] = ref_batch['img'].float().to(device) #change this

    ref_batch['mask'] = torch.from_numpy(ref_batch['mask']).to(device)
    
    assert ref_batch['img'].shape[0] == len(ref_batch['loc']), 'Number of frames and locations do not match'

    im_q = transforms.functional.to_tensor(im_q).unsqueeze(0)
    q_batch = im_q.expand((ref_batch['img'].shape[0], 1, imsize, imsize)).to(device)

    #print('shape win', ref_batch['img'].shape)
    #print('loc win', len(ref_batch['loc']))
    torch.cuda.empty_cache()
    return q_batch, ref_batch, overflow, batch_info