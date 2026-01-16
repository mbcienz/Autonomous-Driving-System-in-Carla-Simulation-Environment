#!/bin/bash
clear

# Kill Python processes related to run_test or server_http, except bash and the script itself
ps -ef | grep -E "python|run_test|server_http" | grep -v "grep" | grep -v "bash" | grep -v "start_all.sh" | awk '{print $2}' | xargs -r kill -9

# Delete old output and result files, if they exist
rm -f ./results/simulation_results.json
rm -f ./results/speed.txt
rm -f ./results/speedplot.png
rm -rf ./__pycache__
rm -f ./log/output_run_test.log
rm -f ./log/output_server.log
echo "Old files deleted successfully."