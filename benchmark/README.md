## Benchmark in Docker Environment

### Pre-built Docker Images
We provide two base Docker images `quay.io/assetopsbench-basic` and `quay.io/assetopsbench-extra` which serves as the AssetOpsBench environment. The base images provide
- `quay.io/assetopsbench-basic`: A minimal Python 3.12 environment with only essential Python packages installed. Suitable for lightweight benchmarking tasks or custom setup.
- `quay.io/assetopsbench-extra`: A full-featured Python 3.12 environment with all components pre-installed, including Individual agents (e.g., IoT Agent), MetaAgent, and AgentHive.

The python environment in both docker images use Conda to manage. The name of pre-built Conda environment is `assetopsbench`. For full list of installed packages, please refer to `basic_requirements.txt` and `extra_requirements.txt`.

### CouchDB Setup

To streamline access to asset data during benchmarking, we use a Dockerized CouchDB instance. This setup allows for easy loading, querying, and management of benchmark datasets within a consistent environment.

A sample `docker-compose.yml` is provided to help you start CouchDB locally with default configuration and credentials. We use `couchdb_setup.sh` as starting command to create a CouchDB database populating with a sample Chiller dataset.


### Run Benchmark Script
In `docker-compose.yml`, we mount the benchmark scripts (`entrypoint.sh` and `run_plan_execute.py`) and scenario files as volumes to assetopsbench container.

Now we run the following command,

```commandline
cd /path/to/AssetOpsBench 
docker-compose -f benchmark/docker-compose.yml build
docker-compose -f benchmark/docker-compose.yml up
```

You can use `.env` or add your environment variables to `.env` which can be passed to containers through `docker-compose.yml`.

 Note that in `entrypoint.sh`, we activate the python environment by `conda activate assetopsbench` first.
 
### Customization

#### Customize Your Python Environment
You can install additional packages (e.g. your own agents) by using `quay.io/assetopsbench-basic` or `quay.io/assetopsbench-extra` as base image. To build your own image, here is an example Dockerfile,
```commandline
FROM quay.io/assetopsbench-basic

ARG CONDA_ENV_NAME=assetopsbench
COPY your_requirements.txt /tmp/your_requirements.txt
RUN opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install /tmp/your_requirements.txt
```

#### Customize Your Run Script
Use your own run script and update `docker-compose.yml`. Recommend mount your scripts as volumes for ease development.
