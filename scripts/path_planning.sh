#!/bin/bash


# We have straight / dijkstra / a_star / rrt / genetic / aco / sa
INIT_METHOD="dijkstra"
# We have penalty / sqp
OPT_METHOD="sqp"
# Set what you want to run
WAYPOINTS=200

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --init_method) INIT_METHOD="$2"; shift ;;
        --opt_method) OPT_METHOD="$2"; shift ;;
        --waypoints) WAYPOINTS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Initialization: $INIT_METHOD"
echo "Optimization: $OPT_METHOD"
echo "Waypoints: $WAYPOINTS"

python main.py \
    --init_method "$INIT_METHOD" \
    --opt_method "$OPT_METHOD" \
    --waypoints "$WAYPOINTS"

