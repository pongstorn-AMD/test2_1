#!/bin/bash

applications=("bfs")
graph_names=("amazon")
#applications=("bfs" "tc" "cc" "pr")
#graph_names=("test5" "test10" "amazon" "as_skitter" "gplus" "higgs" "livejournal" "orkut" "pokec" "roadNetCA" "twitch" "youtube" "web_berkstan" "web_google" "wiki_talk" "wiki_topcats")
#graph_names=("test10")
OUTPUT_FOLDER="/workdir/ARTIFACTS/test5/"

for application in "${applications[@]}"
do
    for graph_name in "${graph_names[@]}"
    do
        echo "Running $application-$graph_name"
        HOME=/workdir /workdir/gem5_pickle/build/ARM/gem5.opt \
            -re \
            --outdir=$OUTPUT_FOLDER/$application-$graph_name-checkpoint \
            --debug-flags=PickleDevicePrefetcherProgressTracker \
            /workdir/experiments/mthread2/gem5_configurations/save_checkpoint.py \
            --application=$application \
            --graph_name=$graph_name;
    done
done


