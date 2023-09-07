#!/bin/bash

# Loop 10 times
for ((i=1; i<=10; i++)); do
    # Run the Python script
    python3 -m gameplay.codenames  

    sleep 1  # 1 second pause
done
