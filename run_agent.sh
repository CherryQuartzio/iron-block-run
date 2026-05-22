if [ "$1" = "--nv" ]; then
    echo "No vision mode enabled."
    xvfb-run -a /opt/conda/envs/minerl/bin/python agent.py
else
  echo "Running with vision mode enabled."
  /opt/conda/envs/minerl/bin/python agent.py
fi
