if [ "$1" = "--nv" ]; then
    echo "Running without vision mode."
    xvfb-run -a /opt/conda/bin/python agent.py
else
  echo "Running with vision mode enabled."
  /opt/conda/bin/python agent.py
fi