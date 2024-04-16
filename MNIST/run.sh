#!/bin/bash

export LC_ALL=C  # ensure we are working with known locales

declare -r PROGRAM=${0##*/}

# No-attack case
listgar=("CM")   # "loss" "medoid" "average" "no-cooperation"
#for cfg1 in "${listgar[@]}"; do
#  python main_mnist.py --nb-workers 10 --data_distribution 'noniid_pathological' --nb-attackers 0 --gar ${cfg1}
#done

listnbbyz=(9)
listattack=("sign_flip" "label_flip" "arbitrary")     #  "sign_flip" "arbitrary"
# attack scenario
for cfg1 in "${listnbbyz[@]}"; do
  for cfg2 in "${listattack[@]}"; do
    for cfg3 in "${listgar[@]}"; do
	    python main_mnist.py --nb-workers 10 --data_distribution 'niid_pathological' --nb-attackers ${cfg1} --attack ${cfg2} --gar ${cfg3}
	  done
	done
done