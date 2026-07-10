#!/bin/bash

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ============================================================
# RUN ONE INDEPENDENT CAMERA VIEW
# ============================================================

run_view () {

    CASE=$1
    VIEW=$2
    VIDEO=$3

    OUT="results/baseline/${CASE}/${VIEW}"

    mkdir -p "$OUT"

    PREFIX="${CASE}_${VIEW}"


    echo
    echo "============================================================"
    echo "CASE : $CASE"
    echo "VIEW : $VIEW"
    echo "VIDEO: $VIDEO"
    echo "OUT  : $OUT"
    echo "============================================================"


    if [ ! -f "$VIDEO" ]; then

        echo
        echo "ERROR: VIDEO NOT FOUND"
        echo "$VIDEO"
        echo

        exit 1

    fi


    # ========================================================
    # 1. OBJECT TRACKING
    # ========================================================

    echo
    echo "[1/5] OBJECT TRACKING"
    echo

    python object_tracker_v6.py \
        "$VIDEO" \
        --output_json \
        "$OUT/${PREFIX}_tracking.json" \
        --output_video \
        "$OUT/${PREFIX}_tracked.mp4"


    # ========================================================
    # 2. VISUAL IDENTITY
    # ========================================================

    echo
    echo "[2/5] VISUAL IDENTITY"
    echo

    python vehicle_identity_v3.py \
        "$VIDEO" \
        "$OUT/${PREFIX}_tracking.json" \
        --output_json \
        "$OUT/${PREFIX}_identity.json"


    # ========================================================
    # 3. OBJECT MOTION ANALYSIS
    # ========================================================

    echo
    echo "[3/5] OBJECT ANALYSIS"
    echo

    python object_analyzer_v4.py \
        "$OUT/${PREFIX}_tracking.json" \
        --identity_json \
        "$OUT/${PREFIX}_identity.json" \
        --output \
        "$OUT/${PREFIX}_object_analysis.json"


    # ========================================================
    # 4. PAIRWISE INTERACTION ANALYSIS
    # ========================================================

    echo
    echo "[4/5] INTERACTION ANALYSIS"
    echo

    python interaction_analyzer.py \
        "$OUT/${PREFIX}_object_analysis.json" \
        --output \
        "$OUT/${PREFIX}_interaction_analysis.json"


    # ========================================================
    # 5. TEMPORAL ABNORMALITY ANALYSIS
    # ========================================================

    echo
    echo "[5/5] TEMPORAL ANALYSIS"
    echo

    python temporal_analyzer.py \
        "$OUT/${PREFIX}_object_analysis.json" \
        --output \
        "$OUT/${PREFIX}_temporal_analysis.json"


    echo
    echo "============================================================"
    echo "COMPLETED"
    echo "$CASE / $VIEW"
    echo "============================================================"
}


# ============================================================
# N1 — NORMAL
# ============================================================

run_view \
    "N1" \
    "INFRA" \
    "videos/N1/N1_infrastructure.mp4"

run_view \
    "N1" \
    "DRONE" \
    "videos/N1/N1_drone.mp4"


# ============================================================
# N2 — NORMAL
# ============================================================

run_view \
    "N2" \
    "INFRA" \
    "videos/N2/N2_infrastructure.mp4"

run_view \
    "N2" \
    "DRONE" \
    "videos/N2/N2_drone.mp4"


# ============================================================
# A1 — ACCIDENT
# ============================================================

run_view \
    "A1" \
    "INFRA" \
    "videos/A1/A1_INFRA_FRONT.mp4"

run_view \
    "A1" \
    "EGO" \
    "videos/A1/A1_EGO_FRONT.mp4"

run_view \
    "A1" \
    "OTHER" \
    "videos/A1/A1_OTHER_FRONT.mp4"


# ============================================================
# A2 — ACCIDENT
# ============================================================

run_view \
    "A2" \
    "INFRA" \
    "videos/A2/A2_INFRA_FRONT.mp4"

run_view \
    "A2" \
    "EGO" \
    "videos/A2/A2_EGO_FRONT.mp4"

run_view \
    "A2" \
    "OTHER" \
    "videos/A2/A2_OTHER_FRONT.mp4"


echo
echo "============================================================"
echo "PER-VIEW BASELINE COMPLETE"
echo "============================================================"
