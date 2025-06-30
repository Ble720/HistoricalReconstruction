import os
import yaml
import numpy as np
import torch
import argparse
import pandas as pd
from match_utils import get_paths, create_save, write_csv

import sys
sys.path.append('../run')
from aspanformer import ASpanFormer
from dataloader import Dataloader

import datetime



def max_scores(counts, confs, ids, loc):
    last = ids[-1] + 1
    size = last - ids[0]
    max_counts = torch.zeros(size)
    max_confs = torch.zeros(size)
    new_loc = []

    #print('bsize', len(counts))
    if len(counts) != len(loc):
        print(len(counts))
        print(len(loc))
    #print('loc shape', loc)
    #print('ids', ids)
    #print('counts shape', counts.shape)
    #print(loc)
    
    loop_ids = sorted(list(set(ids)))
    #print('set ids', loop_ids)
    for idx, i in enumerate(loop_ids):
        indexes = torch.where(torch.tensor(ids) == i)
        #print('indexes', loc[indexes[0]])
        new_loc.append(loc[indexes[0]][torch.argmax(counts[indexes]).item()])
        max_counts[idx] = max(counts[indexes])
        max_confs[idx] = max(confs[indexes])
    #print(';newloc', new_loc)
    return max_counts, max_confs, new_loc



def match_windows(m_counts, m_confs, m_ids, frags, mode='max', q_ratio=None):
    batchsize = len(m_counts) // frags
    new_counts = torch.zeros(batchsize)
    new_confs = torch.zeros(batchsize)
    if mode == 'max':
        for i in range(batchsize):
            new_counts[i] = sum(m_counts[i:(i+1)*frags])/frags
            new_confs[i] = sum(m_confs[i:(i+1)*frags])/frags
    elif mode == 'ratio' and q_ratio is not None:
        rc = int(round(frags**1/2))
        m_counts = m_counts.reshape()
        
        rc_id = [(r,c) for r in range(rc) for c in range(rc)]
    
    return new_counts, new_confs


def match(source, target, topk, save, batch_size, resume):
    create_save(save)
    spath = os.listdir(source)

    #Get only source iamges that are in the label list
    df = pd.read_csv('./key/label_clean.csv', header=0)
    spath = df['Query'].drop_duplicates().tolist()

    spath = [source + '/' + p for p in spath]
    config_file = '../configs/aspanformer.yml'
    with open(config_file, 'r') as f:

        args = yaml.load(f, Loader=yaml.FullLoader)['example']
        if 'ckpt' in args:
            args['ckpt'] = os.path.join('..', args['ckpt'])

    # Init model
    model = ASpanFormer(args)
    matcher = lambda ipath1, ipath2, indexes, batchsize, overflow: model.match_windows(ipath1, ipath2, indexes, batchsize, overflow)

    ref_paths = Dataloader(target, batch_size)
    tpath = ref_paths.im_paths()
    
    # print(target_iter)
    
    for i, s in enumerate(spath):
        score = torch.zeros(len(tpath))
        conf_score = torch.zeros(len(tpath))

        if i < resume:
            continue
        
        iter_ref_paths = iter(ref_paths)

        overflow = fine_overflow = None
        incomplete = {}
        incomplete['counts'] = incomplete['confs'] = None
        incomplete['id'] = []
        incomplete['loc'] = []
        incomplete['path'] = []


        for r_index, refs in iter_ref_paths:
            match, conf, batch_info, ids, loc, overflow = matcher(s, refs, r_index, batch_size, overflow)

            if overflow['index'] != -1:
                iter_ref_paths.overflow(overflow['index'] + 1)

            start = ids[0]
            if overflow['index'] == -1 or overflow['index'] not in ids:
                end = ids[-1] + 1
                split = None
            else:
                end = overflow['index']
                split = ids.index(overflow['index'])

            #print('new coarse')
            counts = torch.bincount(match, minlength=batch_info['size']).cpu()
            confs = torch.bincount(match, weights=torch.tensor(conf).cuda(), minlength=batch_info['size']).cpu()

            #print(match)
            #print(counts)


            if split != 0:
                curr_counts = counts[:split] if incomplete['counts'] is None else torch.cat((incomplete['counts'], counts[:split]))
                curr_confs = confs[:split] if incomplete['confs'] is None else torch.cat((incomplete['confs'], confs[:split]))
                curr_ids = incomplete['id'] + ids[:split]
                curr_loc = incomplete['loc'] + loc[:split]

                max_counts, max_confs, max_loc = max_scores(curr_counts, curr_confs, curr_ids, np.array(curr_loc))

                #print(len(max_loc), max_counts.shape, max_confs.shape)
                '''
                cid = curr_ids[0]
                for nloc in max_loc:
                    if nloc.tolist() not in batch_info[cid]['loc']:
                        print(cid, nloc, batch_info[cid]['loc'])
                    cid += 1
                '''
                incomplete['counts'] = None if split is None else counts[split:]
                incomplete['confs'] = None if split is None else confs[split:]
                incomplete['id'] = [] if split is None else ids[split:]
                incomplete['loc'] = [] if split is None else loc[split:]
            else:
                incomplete['counts'] = counts[split:] if incomplete['counts'] is None else torch.cat((incomplete['counts'], counts[split:]))
                incomplete['confs'] = confs[split:] if incomplete['confs'] is None else torch.cat((incomplete['confs'], confs[split:]))
                incomplete['id'] += ids[split:]
                incomplete['loc'] += loc[split:]

            #print('incomplete', incomplete['counts'].shape, len(incomplete['loc']))

            score[start:end] = torch.max(max_counts, score[start:end])
            conf_score[start:end] = torch.max(max_confs, conf_score[start:end])

            '''
            fine_r_index = list(range(curr_ids[0], curr_ids[0]+len(max_loc)))
            fine_ref = incomplete['path'] + refs[:len(fine_r_index)-len(incomplete['path'])]
            fine_result = fine_matcher(s, fine_ref, fine_r_index, batch_size, max_loc)

            score[start:end] = torch.max(fine_result['counts'], score[start:end])
            conf_score[start:end] = torch.max(fine_result['confs'], conf_score[start:end])
            
            incomplete['path'] = overflow['path']
            '''

        print(f'Calc top {topk}')

        qnames = [spath[i].replace('\\', '/').split('/')[-1]]*topk

        weighted_score = score + conf_score
        max_wsc, max_arg = torch.topk(weighted_score, topk, largest=True)  
        score1 = max_wsc.tolist()
        score2 = score[max_arg].tolist() 
        rnames = [tpath[r.item()] for r in max_arg]
        write_csv(qnames, rnames, score1, score2, 'W', save)

        max_sc, max_arg = torch.topk(score, topk, largest=True)  
        score1 = max_sc.tolist()
        score2 = conf_score[max_arg].tolist()
        #score3 = thresh_score[max_arg].tolist()
        rnames = [tpath[r.item()] for r in max_arg] #.replace('\\', '/').split('/')[-1]
        write_csv(qnames, rnames, score1, score2, 'M', save)

        max_sc, max_arg = torch.topk(conf_score, topk, largest=True)  
        score1 = max_sc.tolist()
        score2 = score[max_arg].tolist()  
        #score3 = thresh_score[max_arg].tolist()
        rnames = [tpath[r.item()] for r in max_arg] #.replace('\\', '/').split('/')[-1]
        write_csv(qnames, rnames, score1, score2, 'CS', save)

        '''
        max, max_arg = torch.topk(thresh_score, topk, largest=True)  
        score1 = max.tolist()
        score2 = score[max_arg].tolist()  
        score3 = conf_score[max_arg].tolist()
        rnames = [tpath[r.item()].replace('\\', '/').split('/')[-1] for r in max_arg]
        write_csv(qnames, rnames, score1, score2, score3, 'TM', save)


        
        max, max_arg = torch.topk(score, topk, largest=True)

        si_name = spath[i].replace('\\', '/').split('/')[-1]

        dir_name = '{}/{}'.format(save, si_name[:-4])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        shutil.copyfile(spath[i], '{}/{}/SRC_{}'.format(save, si_name[:-4], si_name))

        for ip in max_arg:
            ti_name = tpath[ip.item()].replace('\\', '/').split('/')[-1]
            shutil.copyfile(tpath[ip.item()], '{}/{}/{}'.format(save, si_name[:-4], ti_name))
        '''
        print(f"source {i+1}/{len(spath)} done\n")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--source', type=str, default='../datasets/original_clahe') #../dist_datasets/ldm_4
    #parser.add_argument('--source', type=str, default='../../EfficientLoFTR/datasets/deqian1') #./../EfficientLoFTR/datasets/src_x4deblur_v2
    parser.add_argument('--target', type=str, default='../datasets/target')
    #parser.add_argument('--target', type=str, default='../../EfficientLoFTR/datasets/ground_truth')
    parser.add_argument('--save', type=str, default='./final_results/original_clahe')#'./rectified_1')
    parser.add_argument('--description', type=str, default="")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--resume', type=int, default=-1)
    opt = parser.parse_args()

    if not opt.source:
        raise Exception("No source path provided")

    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    if not opt.target or not opt.save:
        raise Exception("No target or save path provided")

    with open('logs.txt', 'a') as f:
        now = datetime.datetime.now()
        f.write('---------------------------------------------------------------\n')
        f.write(now.strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(f'source path: {opt.source}\n')
        f.write(f'target path: {opt.target}\n')
        f.write(f'save path: {opt.save}\n')
        f.write(f'description: {opt.description}\n')

    match(opt.source, opt.target, opt.topk, opt.save, opt.batch_size, opt.resume)
