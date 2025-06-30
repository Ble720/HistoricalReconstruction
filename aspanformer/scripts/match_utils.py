import os, csv
import torch
import numpy as np

def cosine_sim(v1, v2, b_ids, batch_len):
    cos_sim = torch.zeros(batch_len)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for i in range(batch_len):
        v1_i = v1[b_ids == i].flatten()
        v2_i = v2[b_ids == i].flatten()
        cos_sim[i] = cos(v1_i,v2_i)
         
    return cos_sim

def get_paths(dir_path):
    all_path = []
    for fRoot, fDirs, fFiles in os.walk(dir_path):
        for ffile in fFiles:
            if not ffile.startswith('.'):
                full_path = os.path.join(fRoot, ffile).replace('/', os.sep)
                all_path.append(full_path)
    all_path = np.array(all_path)
    print(len(all_path))
    return all_path

def create_save(save):
    if not os.path.exists(save):
        os.makedirs(save)
        
    with open(f'{save}/predict_M.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matches', 'Confidence_Sum','Query','Reference'])

    with open(f'{save}/predict_W.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Weighted_Score', 'Matches','Query','Reference'])

    with open(f'{save}/predict_CS.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Confidence_Sum','Matches','Query','Reference'])

def create_save_sim(save):
    if not os.path.exists(save):
        os.makedirs(save)
        
        with open(f'{save}/predict_cossim.csv', 'a', newline='') as csvfile:
            print('create save')
            writer = csv.writer(csvfile)
            writer.writerow(['Cossim','Query','Reference'])


def write_csv_sim(query_names, ref_names, cos_sim, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f'{save_path}/predict_cossim.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        num_pairs = len(cos_sim)
        for i in range(num_pairs):
            rname = ref_names[i].replace('\\', '/').split('/')[-1]
            qname = query_names[i].replace('\\', '/').split('/')[-1]
            writer.writerow([cos_sim[i], qname, rname])


def write_csv(query_names, ref_names, score1, score2, mode, save_path):
    with open(f'{save_path}/predict_{mode}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        num_pairs = len(score1)
        for i in range(num_pairs):
            rname = ref_names[i].replace('\\', '/').split('/')[-1]
            qname = query_names[i].replace('\\', '/').split('/')[-1]
            writer.writerow([score1[i], score2[i], qname, rname])