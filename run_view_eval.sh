#!/bin/bash

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


run_case () {

    CASE=$1
    VIDEO=$2

    OUT="results/baseline/${CASE}"

    mkdir -p "$OUT"

    echo
    echo "============================================================"
    echo "BASELINE CASE : $CASE"
    echo "VIDEO         : $VIDEO"
    echo "OUTPUT        : $OUT"
    echo "============================================================"


    echo
    echo "[1/6] OBJECT TRACKING"
    echo

    python object_tracker_v6.py \
        "$VIDEO" \
        --output_json "$OUT/${CASE}_tracking.json" \
        --output_video "$OUT/${CASE}_tracked.mp4"


    echo
    echo "[2/6] VISUAL IDENTITY"
    echo

    python vehicle_identity_v3.py \
        "$VIDEO" \
        "$OUT/${CASE}_tracking.json" \
        --output_json "$OUT/${CASE}_identity.json"


    echo
    echo "[3/6] OBJECT ANALYSIS"
    echo

    python object_analyzer_v4.py \
        "$OUT/${CASE}_tracking.json" \
        --identity_json "$OUT/${CASE}_identity.json" \
        --output "$OUT/${CASE}_object_analysis.json"


    echo
    echo "[4/6] INTERACTION ANALYSIS"
    echo

    python interaction_analyzer.py \
        "$OUT/${CASE}_object_analysis.json" \
        --output "$OUT/${CASE}_interaction_analysis.json"


    echo
    echo "[5/6] TEMPORAL ANALYSIS"
    echo

    python temporal_analyzer.py \
        "$OUT/${CASE}_object_analysis.json" \
        --output "$OUT/${CASE}_temporal_analysis.json"


    echo
    echo "[6/6] EVENT VERIFICATION"
    echo

    python event_verifier.py \
        "$VIDEO" \
        "$OUT/${CASE}_temporal_analysis.json" \
        --output "$OUT/${CASE}_event_verification.json"


    echo
    echo "============================================================"
    echo "COMPLETED : $CASE"
    echo "============================================================"
}


run_case \
    "N1" \
    "videos/N1/N1_MULTIVIEW.mp4"

run_case \
    "N2" \
    "videos/N2/N2_MULTIVIEW.mp4"

run_case \
    "A1" \
    "videos/A1/A1_MULTIVIEW.mp4"

run_case \
    "A2" \
    "videos/A2/A2_MULTIVIEW.mp4"


echo
echo "============================================================"
echo "FOUR-CASE BASELINE COMPLETE"
echo "============================================================"
