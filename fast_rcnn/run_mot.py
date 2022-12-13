#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import csv
import logging
import os
import os.path as osp
from glob import glob

import numpy as np
import yaml
from matplotlib.pyplot import imread
from tqdm import tqdm

from head_detection.data import cfg_mnet, cfg_res50_4fpn, cfg_res152
from obj_detect import HeadHunter
from tracker import Tracker

from collections import defaultdict

# python3 run_mot.py --base_dir /workspace/HeadHunter--T/HT21 --save_path /workspace/HeadHunter--T/save_dir
# python3 run_mot.py --base_dir /workspace/HeadHunter--T/HT21 --save_path /workspace/HeadHunter--T/save_practice_dir
# /workspace/HeadHunter--T/save_dir/HT-test/HT21-11
np.random.seed(seed=12345)

Parser = argparse.ArgumentParser(description='Testing the tracker on MOT style')
Parser.add_argument('--base_dir',
                     default = '/workspace/juntae/Crowd-Estimation/fast_rcnn/HT21',
                    type=str, help='Base directory for the dataset')
Parser.add_argument('--save_path',
                    default = '/workspace/juntae/Crowd-Estimation/fast_rcnn/run_save',
                    type=str, help='Directory to save the results')
Parser.add_argument('--cfg_file', 
                    default='./config/config.yaml',
                    type=str, help='path to config file')
Parser.add_argument('--start_ind',
                    default=0,
                    type=int, help='should I skip any seq?')
Parser.add_argument('--save_frames', 
                    # default=False,
                    default = '/workspace/HeadHunter--T/save_frame',
                    type=bool, help='should I save frames?')

# practice로 하면, 실제 사용할 데이터 사용
Parser.add_argument('--dataset', 
                    # default='all',
                    default = 'practice',
                    type=str, help='Train/Test/All')

Parser.add_argument('--detector', 
                    default='det',
                    type=str, help='Directory where public detection are saved')


args = Parser.parse_args()
log = logging.getLogger('Head Tracker on MOT style data')
log.setLevel(logging.DEBUG)


# Get parameters from Config file
with open(args.cfg_file, 'r') as stream:
    CONFIG = yaml.safe_load(stream)

det_cfg = CONFIG['DET']['det_cfg']
backbone = CONFIG['DET']['backbone']
tracktor_cfg = CONFIG['TRACKTOR']
motion_cfg = CONFIG['MOTION']
tracker_cfg = CONFIG['TRACKER']
gen_cfg = CONFIG['GEN']
# is_save = gen_cfg['save_frames']

# Initialise network configurations
if backbone == 'resnet50':
    net_cfg = cfg_res50_4fpn
elif backbone == 'resnet152':
    net_cfg = cfg_res152
elif backbone == 'mobilenet':
    net_cfg = cfg_mnet
else:
    raise ValueError("Invalid Backbone")


def read_public_det(det):
    det_dict = defaultdict(list)
    with open(det, 'r') as dfile:
        for i in dfile.readlines():
            cur_det = [float(k) for k in i.strip('\n').split(',')]
            det_dict[int(cur_det[0])].append([cur_det[2],
                                              cur_det[3],
                                              cur_det[4]+cur_det[2],
                                              cur_det[3]+cur_det[5],
                                              cur_det[6]/100.])
    return det_dict


# Get sequences of MOT Dataset
# datasets = ('HT-train', 'HT-test') if args.dataset == 'all' else (args.dataset, )
# datasets = ('HT-practice', ) if args.dataset == 'practice' else ('HT-train', 'HT-test')
if args.dataset == 'practice2':
    datasets =  ('HT-practice2',)
elif args.dataset == 'practice':
    datasets = ('HT-practice', )
elif args.dataset == 'all':
    datasets = ('HT-train', 'HT-test')

for dset in datasets:
    mot_dir = osp.join(args.base_dir, dset)  # /workspace/HeadHunter--T/HT21/HT-train(test)
    print(mot_dir)
    mot_seq = os.listdir(mot_dir)[args.start_ind:]   # [HT21-11, 12, 13, 14, 15]
    mot_paths = sorted([osp.join(mot_dir, seq) for seq in mot_seq])   # [/workspace/HeadHunter--T/HT21/HT-train(test)/HT21-11, 12, 13, 14, 15]

    # Create the required saving directories
    if args.save_frames:
        save_paths = [osp.join(args.save_path, seq) for seq in mot_seq]
        _ = [os.makedirs(i, exist_ok=True) for i in save_paths]
        assert len(mot_paths) == len(save_paths)

    all_results = []


    for ind, mot_p in enumerate(tqdm(mot_paths)):  # osp.join(mot_p, 'A') == /workspace/HeadHunter--T/HT21/HT-train(test)/A
        seqfile = osp.join(mot_p, 'seqinfo.ini')  # seqfile = /workspace/HeadHunter--T/HT21/HT-train(test)/HT21-##/seqinfo.ini

        config = configparser.ConfigParser()
        config.read(seqfile)
        c_width = int(config['Sequence']['imWidth'])
        c_height = int(config['Sequence']['imHeight'])
        seq_length = int(config['Sequence']['seqLength'])
        seq_ext = config['Sequence']['imExt']
        seq_dir = config['Sequence']['imDir']
        cam_motion = bool(config['Sequence'].get('cam_motion', False))
        seq_name = config['Sequence']['name']

        traj_dir = osp.join(args.save_path, dset, mot_seq[ind])  # traj_dir = /workspace/HeadHunter--T/save_dir/HT-test(train)/HT21-##
        os.makedirs(traj_dir, exist_ok=True)
        traj_fname = osp.join(traj_dir, 'pred.txt')
        log.info("Total length is " + str(seq_length))
        print('total lenght ', str(seq_length))

        im_shape = (c_height, c_width, 3)
        im_path = osp.join(mot_p, seq_dir)
        seq_images = sorted(glob(osp.join(im_path, '*'+seq_ext)))
        # print(im_path, '*'+seq_ext)

        # Create detector and traktor
        detector = HeadHunter(net_cfg, det_cfg).cuda()
        save_dir = save_paths[ind] if args.save_frames else None

        """
        실제 파일을 사용할 경우, det.txt가 없을 확률이 매우 높다.
        이를 위해서, config의 use_public을 False로 설정할 필요 있음.
        """
        # Read Public detection if necessary
        if tracker_cfg['use_public'] is True:
            print("using " + args.detector)
            det_file = args.detector + '.txt'   # det_file = det.txt
            det_dict = read_public_det(osp.join(mot_p, 'det', det_file))  # det_dict = read_public_det(workspace/HeadHunter--T/HT21/HT-train(test)/HT21-##/det/det.txt)

            tracker = Tracker(detector, tracker_cfg, tracktor_cfg, motion_cfg, im_shape,
                        save_dir=save_dir,
                        save_frames=args.save_frames, cam_motion=cam_motion,
                        public_detections=det_dict)
        else :
            tracker = Tracker(detector, tracker_cfg, tracktor_cfg, motion_cfg, im_shape,
                              save_dir=save_dir,
                              save_frames=args.save_frames, cam_motion=cam_motion,
                              public_detections=None)

        for im0 in tqdm(seq_images[200:300]):
            cur_im = imread(im0)
            #
            # print(im0)
            tracker.step(cur_im)

        cur_result = tracker.get_results()
        with open(traj_fname, "w+") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in cur_result.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame, i, x1, y1, x2-x1+1,
                                    y2-y1+1, 1, 1, 1, 1])
