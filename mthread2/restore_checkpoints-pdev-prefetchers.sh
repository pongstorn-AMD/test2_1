#!/bin/bash

applications=("bfs")
#graph_names=("test10")
graph_names=("amazon")
#private_cache_prefetchers=("imp" "ampm" "stride" "multiv1")
private_cache_prefetchers=("stride")
#OUTPUT_FOLDER="/workdir/ARTIFACTS/results_v8/"
OUTPUT_FOLDER="/workdir/ARTIFACTS/test5/"

PREFETCH_DISTANCE=48
OFFSET=16

for application in "${applications[@]}"
do
    for graph_name in "${graph_names[@]}"
    do
        for private_cache_prefetcher in "${private_cache_prefetchers[@]}"
        do
            echo "Running $application-$graph_name with $private_cache_prefetcher"
            /workdir/gem5_pickle/build/ARM/gem5.opt \
                -re \
                --outdir=$OUTPUT_FOLDER/$application-$graph_name-pdev_distance_${PREFETCH_DISTANCE}_offset_$OFFSET-$private_cache_prefetcher \
                --debug-flags=PickleDeviceUncacheableForwarding \
                 /workdir/experiments/mthread2/gem5_configurations/restore_checkpoint.py \
                --application $application \
                --graph_name=$graph_name \
		--prefetch_mode "single_prefetch" \
            	--bulk_prefetch_chunk_size 10 \
                --pickle_cache_size 256KiB \
                --prefetch_drop_distance 0 \
                --delegate_last_layer_prefetch False \
                --concurrent_work_item_capacity 64 \
            	--bulk_prefetch_num_prefetches_per_hint 5 \
                --enable_pdev=True \
                --prefetch_distance=$PREFETCH_DISTANCE \
                --offset_from_pf_hint=$OFFSET \
                --pdev_num_tbes 1024 \
                --private_cache_prefetcher=$private_cache_prefetcher &
        done
    done
done
