#!/bin/bash

agent_file=$(find agent -maxdepth 1 -type f | head -n 1)
saved_file=$(find saved_agents -maxdepth 1 -type f | head -n 1)

if [[ -z "$agent_file" || -z "$saved_file" ]]; then
    echo "Missing file in agent or saved_agents"
    exit 1
fi

mv "$agent_file" agent_archive/
mv "$saved_file" agent/

echo "Agent file moved to agent_archive/"
echo "Saved file moved to agent/"