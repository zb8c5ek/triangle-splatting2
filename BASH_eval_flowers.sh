#!/bin/bash
#SBATCH --output=logs/eval_flowers.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G # Memory to allocate in MB per allocated CPU core
#SBATCH --time="0-00:50:00" # Max execution time
#SBATCH --account=telim


#micromamba activate pow3r

python train.py \
-s /d_disk/triangle-splatting2/assets/360_extra_scenes/flowers  \
-i images_4 -m models/$1/flowers --eval


python render.py --iteration 30000 \
-s /d_disk/triangle-splatting2/assets/360_extra_scenes/flowers \
-m /d_disk/triangle-splatting2/models/flowers \
--eval --skip_train --quiet

python metrics.py -m /d_disk/triangle-splatting2/models/flowers

python create_ply.py /d_disk/triangle-splatting2/models/flowers/point_cloud/iteration_30000
python create_video.py -m /d_disk/triangle-splatting2/models/flowers -s /d_disk/triangle-splatting2/assets/360_extra_scenes/flowers