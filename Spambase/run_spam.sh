#!/bin/bash

export LC_ALL=C  # ensure we are working with known locales

declare -r PROGRAM=${0##*/}

# No-attack case
listgar=("medoid")   # "loss" "krum" "no-coop" "medoid" "average" "CM" "TM"
#for cfg1 in "${listgar[@]}"; do
#  python main_spam.py --nb-workers 10 --nb-attackers 0 --data_distribution 'niid_practical' --model 'MLP_1layer'  --gar ${cfg1}
#done

##'niid_practical'  niid_pathological
listnbbyz=(3)
listattack=("arbitrary")   # "empire" "little" "arbitrary"
# attack scenario
for cfg1 in "${listnbbyz[@]}"; do
	echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${cfg2}"
  for cfg2 in "${listattack[@]}"; do
    for cfg3 in "${listgar[@]}"; do
	    python main_spam.py --nb-workers 10 --data_distribution 'niid_practical' --model 'MLP_1layer' --nb-attackers ${cfg1} --attack ${cfg2} --gar ${cfg3}
	  done
	done
done