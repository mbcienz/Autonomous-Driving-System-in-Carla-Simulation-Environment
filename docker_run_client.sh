docker run --rm \
        -v $(pwd)/userCode/:/workspace/team_code/ \
        --network=bridge \
        --name carla-client-instance-${USER} \
        -p 8818:8818\
        -p 9818:9818 \
        -d\
        -it carla-client \
        /bin/bash
