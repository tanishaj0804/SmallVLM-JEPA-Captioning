#!/bin/bash

set -e


run_case () {

    CASE=$1
    shift

    FUSION_OUT="results/baseline/${CASE}/${CASE}_multiview_fusion.json"

    CRITICALITY_OUT="results/baseline/${CASE}/${CASE}_criticality_analysis.json"


    echo
    echo "============================================================"
    echo "CASE : $CASE"
    echo "============================================================"


    echo
    echo "[1/2] MULTIVIEW EVENT FUSION V3"
    echo

    python multiview_event_fusion.py \
        --case "$CASE" \
        "$@" \
        --output "$FUSION_OUT"


    echo
    echo "[2/2] CRITICALITY ANALYSIS"
    echo

    python criticality_analyzer.py \
        "$FUSION_OUT" \
        --output "$CRITICALITY_OUT"


    echo
    echo "============================================================"
    echo "COMPLETED : $CASE"
    echo "============================================================"
}


# ============================================================
# N1
# ============================================================

run_case \
    "N1" \
    --view "INFRA=results/baseline/N1/INFRA/N1_INFRA_interaction_analysis.json,results/baseline/N1/INFRA/N1_INFRA_temporal_analysis.json" \
    --view "DRONE=results/baseline/N1/DRONE/N1_DRONE_interaction_analysis.json,results/baseline/N1/DRONE/N1_DRONE_temporal_analysis.json"


# ============================================================
# N2
# ============================================================

run_case \
    "N2" \
    --view "INFRA=results/baseline/N2/INFRA/N2_INFRA_interaction_analysis.json,results/baseline/N2/INFRA/N2_INFRA_temporal_analysis.json" \
    --view "DRONE=results/baseline/N2/DRONE/N2_DRONE_interaction_analysis.json,results/baseline/N2/DRONE/N2_DRONE_temporal_analysis.json"


# ============================================================
# A1
# ============================================================

run_case \
    "A1" \
    --view "INFRA=results/baseline/A1/INFRA/A1_INFRA_interaction_analysis.json,results/baseline/A1/INFRA/A1_INFRA_temporal_analysis.json" \
    --view "EGO=results/baseline/A1/EGO/A1_EGO_interaction_analysis.json,results/baseline/A1/EGO/A1_EGO_temporal_analysis.json" \
    --view "OTHER=results/baseline/A1/OTHER/A1_OTHER_interaction_analysis.json,results/baseline/A1/OTHER/A1_OTHER_temporal_analysis.json"


# ============================================================
# A2
# ============================================================

run_case \
    "A2" \
    --view "INFRA=results/baseline/A2/INFRA/A2_INFRA_interaction_analysis.json,results/baseline/A2/INFRA/A2_INFRA_temporal_analysis.json" \
    --view "EGO=results/baseline/A2/EGO/A2_EGO_interaction_analysis.json,results/baseline/A2/EGO/A2_EGO_temporal_analysis.json" \
    --view "OTHER=results/baseline/A2/OTHER/A2_OTHER_interaction_analysis.json,results/baseline/A2/OTHER/A2_OTHER_temporal_analysis.json"


echo
echo "============================================================"
echo "FOUR-CASE CRITICALITY EVALUATION COMPLETE"
echo "============================================================"
