#!/bin/bash

export LC_ALL=C  # ensure we are working with known locales

declare -r PROGRAM=${0##*/}

# No-attack case
listgar=("loss" "krum" "CM" "TM" "medoid" "average")   # "loss" "krum" "no-coop" "medoid" "average" "CM"
#for cfg1 in "${listgar[@]}"; do
#  python main_har.py --nb-workers 30 --nb-attackers 13 --attack "little" --gar ${cfg1}
#done


listnbbyz=(6)    # 29
listattack=("empire" "little" "label_flip")
# attack scenario
for cfg1 in "${listnbbyz[@]}"; do
	echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${cfg2}"
  for cfg2 in "${listattack[@]}"; do
    for cfg3 in "${listgar[@]}"; do
	    python main_har.py --nb-workers 30 --nb-attackers ${cfg1} --attack ${cfg2} --gar ${cfg3}
	  done
	done
done