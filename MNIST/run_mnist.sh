#!/bin/bash

export LC_ALL=C  # ensure we are working with known locales

declare -r PROGRAM=${0##*/}

listgar=("loss" "krum" "medoid" "median" "average" "TM")   # "no-coop" "krum" "loss" "medoid" "median"
for cfg1 in "${listgar[@]}"; do
  python main_mnist.py --nb-workers 10 --data_distribution 'niid_pathological' --model 'conv' --nb-attackers 0 --gar ${cfg1}
done

#listnbbyz=(3)
#listattack=("sign_flip" "arbitrary"  "empire" "little" "label_flip")     #  "sign_flip" "arbitrary"  "empire" "little"
## attack scenario
#for cfg1 in "${listnbbyz[@]}"; do
#  for cfg2 in "${listattack[@]}"; do
#    for cfg3 in "${listgar[@]}"; do
#	    python main_mnist.py --nb-workers 10 --data_distribution 'niid_pathological' --model 'conv' --nb-attackers ${cfg1} --attack ${cfg2} --gar ${cfg3}
#	  done
#	done
#done
#
#for cfg1 in "${listgar[@]}"; do
#  python main_mnist.py --nb-workers 10 --data_distribution 'niid_pathological' --model 'full' --nb-attackers 0 --gar ${cfg1}
#done