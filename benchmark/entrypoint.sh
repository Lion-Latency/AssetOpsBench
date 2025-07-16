#!/bin/bash
# Activate conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

# (Optional) start your Python app here
which python
python --version
python -m pip show agent_hive
python -m pip show reactxen
python -m pip show fmsr_agent
python -m pip show iotagent
python -m pip show tsfmagent

# Keep the container alive
tail -f /dev/null
