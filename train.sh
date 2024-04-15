python -W ignore train.py \
  --dataset LCSD --data-root /Your/path/to/dataset \
  --backbone resnet101 --shot 1 --episode 6000 --snapshot 200

# python -W ignore train.py \
#    --dataset LCSD --data-root /Your/path/to/dataset \
#    --backbone resnet101 --shot 5 --episode 6000 --snapshot 200

# python -W ignore train.py \
#   --dataset llCrackSeg9k --data-root /Your/path/to/dataset \
#   --backbone resnet101 --shot 1 --episode 18000 --snapshot 1200

# python -W ignore train.py \
#   --dataset llCrackSeg9k --data-root /Your/path/to/dataset \
#   --backbone resnet101 --shot 5 --episode 18000 --snapshot 1200
