#!/bin/bash

python3 inference.py \
--test_imgs /studio-storage1/datasets/hdr_test_data/sihdr/linearized/clip_95 \
--output_path test_outputs \
--use_exr --end_id 5