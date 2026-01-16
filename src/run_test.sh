#!/bin/bash
#qdtrack_ training.xml
# export ROUTES=${LEADERBOARD_ROOT}/data/routes_controlling.xml
export ROUTES=/workspace/team_code/carlaContolL6/Progetto/routes/route_controlling.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/workspace/team_code/carlaContolL6/Progetto/basic_autonomous_agent.py
export TEAM_CONFIG=/workspace/team_code/carlaContolL6/Progetto/config_agent_basic.json
export CHECKPOINT_ENDPOINT=${LEADERBOARD_ROOT}/results.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=172.16.174.233
export CARLA_PORT=6018
export CARLA_TRAFFIC_MANAGER_PORT=8818
export CHECKPOINT_ENDPOINT=/workspace/team_code/carlaContolL6/Progetto/results/simulation_results.json
export DEBUG_CHECKPOINT_ENDPOINT=/workspace/team_code/carlaContolL6/Progetto/results/live_results.txt
export RESUME=0
export TIMEOUT=60
# 193.205.163.183
# 193.205.163.17

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--host=${CARLA_HOST} \
--port=${CARLA_PORT} \
--timeout=${TIMEOUT} \
--traffic-manager-port=${CARLA_TRAFFIC_MANAGER_PORT} 
