#!/usr/bin/env bash

declare -i iter=$1
#declare -s file=2

for((i=0;i<$iter;i++))
    do
        ./train.py $2 $2
        file=$(sed -n '4p' ./model/$2)
        echo $file
        cp model/$2 tmpModel/$file
    done
