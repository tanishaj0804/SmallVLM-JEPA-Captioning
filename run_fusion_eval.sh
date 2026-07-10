#!/bin/bash


set -e


echo
echo "============================================================"
echo "N1 MULTIVIEW FUSION"
echo "============================================================"

python multiview_event_fusion.py \
    --case N1 \
    --view "INFRA=results/baseline/N1/INFRA/N1_INFRA_interaction_analysis.json,results/baseline/N1/INFRA/N1_INFRA_temporal_analysis.json" \
    --view "DRONE=results/baseline/N1/DRONE/N1_DRONE_interaction_analysis.json,results/baseline/N1/DRONE/N1_DRONE_temporal_analysis.json" \
    --output results/baseline/N1/N1_multiview_fusion.json


echo
echo "============================================================"
echo "N2 MULTIVIEW FUSION"
echo "============================================================"

python multiview_event_fusion.py \
    --case N2 \
    --view "INFRA=results/baseline/N2/INFRA/N2_INFRA_interaction_analysis.json,results/baseline/N2/INFRA/N2_INFRA_temporal_analysis.json" \
    --view "DRONE=results/baseline/N2/DRONE/N2_DRONE_interaction_analysis.json,results/baseline/N2/DRONE/N2_DRONE_temporal_analysis.json" \
    --output results/baseline/N2/N2_multiview_fusion.json


echo
echo "============================================================"
echo "A1 MULTIVIEW FUSION"
echo "============================================================"

python multiview_event_fusion.py \
    --case A1 \
    --view "INFRA=results/baseline/A1/INFRA/A1_INFRA_interaction_analysis.json,results/baseline/A1/INFRA/A1_INFRA_temporal_analysis.json" \
    --view "EGO=results/baseline/A1/EGO/A1_EGO_interaction_analysis.json,results/baseline/A1/EGO/A1_EGO_temporal_analysis.json" \
    --view "OTHER=results/baseline/A1/OTHER/A1_OTHER_interaction_analysis.json,results/baseline/A1/OTHER/A1_OTHER_temporal_analysis.json" \
    --output results/baseline/A1/A1_multiview_fusion.json


echo
echo "============================================================"
echo "A2 MULTIVIEW FUSION"
echo "============================================================"

python multiview_event_fusion.py \
    --case A2 \
    --view "INFRA=results/baseline/A2/INFRA/A2_INFRA_interaction_analysis.json,results/baseline/A2/INFRA/A2_INFRA_temporal_analysis.json" \
    --view "EGO=results/baseline/A2/EGO/A2_EGO_interaction_analysis.json,results/baseline/A2/EGO/A2_EGO_temporal_analysis.json" \
    --view "OTHER=results/baseline/A2/OTHER/A2_OTHER_interaction_analysis.json,results/baseline/A2/OTHER/A2_OTHER_temporal_analysis.json" \
    --output results/baseline/A2/A2_multiview_fusion.json


echo
echo "============================================================"
echo "FOUR-CASE FUSION EVALUATION COMPLETE"
echo "============================================================"
