#coding:utf-8

import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package  images')
    # general
    parser.add_argument('--dir', default='/home/data/tface/test/data/lfw_cropped2', help='')  #aligned LFW data root dir  
    parser.add_argument('--image-size', type=str, default='112,112', help='') #image size
    parser.add_argument('--output', default='./lfw.bin', help='path to save.') # where to save  
    args = parser.parse_args()
    image_size = [int(x) for x in args.image_size.split(',')]
   
    pairs_path = '/home/data/facenet/data/LFW/pairs.txt' #from LFW download pairs.txt
    pairs = read_pairs(pairs_path)

    img_paths, issame_list = get_paths(args.dir, pairs)

    img_bins = []
    i = 0
    for path in img_paths:
        with open(path, 'rb') as fin:
            _bin = fin.read()
            img_bins.append(_bin)
            i += 1
    with open(args.output, 'wb') as f:
        pickle.dump((img_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
