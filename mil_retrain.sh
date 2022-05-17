#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -A bdlds05
#SBATCH --time=2-00:00:00        # Run for a max of 2 days
#Now run the job
. /nobackup/projects/bdlds05/lucyg/spack/share/spack/setup-env.sh
module add openslide-3.4.1-gcc-8.5.0-3swl63u
source /nobackup/projects/bdlds05/lucyg/miniconda/bin/activate
source activate mil_project
CUDA_VISIBLE_DEVICES=0 python MIL_retrain_2.py --train_lib /nobackup/projects/bdlds05/lucyg/MIL-nature-medicine-2019-master/datasets/high_low/train/train_example_slide_dict.pt --val_lib /nobackup/projects/bdlds05/lucyg/MIL-nature-medicine-2019-master/datasets/high_low/val/val_example_slide_dict.pt --output /nobackup/projects/bdlds05/lucyg/MIL-nature-medicine-2019-master/ --batch_size 512 --nepochs 13 --k 10