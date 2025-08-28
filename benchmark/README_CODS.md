
# AssetOpsBench Setup Guide

This guide describes how to fork, configure, and run the **AssetOpsBench** project using Conda and Docker on macOS.

---

## 1. Fork and Clone

Fork the AssetOpsBench repository:

```bash
git clone https://github.com/IBM/AssetOpsBench.git
```

Switch to the `Asset_CODS` branch:

```bash
git checkout Asset_CODS
```

---

## 2. Project Structure

Your local root directory:

```
/Users/jzhou/work/notebooks/AssetOpsBench
```

Create the folder if it doesn’t exist:

```bash
mkdir -p /Users/jzhou/work/notebooks/AssetOpsBench
```

---

## 3. Conda Environment

Create and activate the Conda environment for **AssetOpsBench**.

Check your Conda installation:

```bash
conda info | grep 'base environment'
```

Example output (Miniforge):

```
base environment : /Users/jzhou/miniforge3  (writable)
```

Adjust your `entrypoint.sh` to source the correct Conda environment:

```bash
# Original (inside Docker image)
source /opt/conda/etc/profile.d/conda.sh

# On macOS with Miniforge
source /Users/jzhou/miniforge3/etc/profile.d/conda.sh
```

---

## 4. Docker Setup

Check your Docker installation:

```bash
which docker
```

Example output:

```
/Users/jzhou/.rd/bin/docker
```

This indicates **Rancher Desktop** is being used:

```bash
ls /Applications | grep -i rancher
# Rancher Desktop.app
```

If you switch to another vendor, reconfigure Docker accordingly.

---

## 5. Build and Run

Follow the official [benchmark README](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/README.md).

### Swap entrypoint scripts (if needed)

```bash
mv entrypoint.sh entrypoint_old.sh
mv entrypoint_joe.sh entrypoint.sh
```

This change accounts for different Conda environment paths.

### Build the Docker image

```bash
docker-compose -f benchmark/docker-compose.yml build
```

### Run

```bash
docker-compose -f benchmark/docker-compose.yml up
```

---

## 6. Troubleshooting

- **Docker not running**

```text
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

➡ Ensure Rancher Desktop (or Docker Desktop) is running.

- **Expired Kubernetes certificate**

Clean up and restart:

```bash
kubectl get nodes
kubectl cluster-info
docker ps
docker images | head
```

- **Segmentation fault**

```text
qemu: uncaught target signal 11 (Segmentation fault) - core dumped
```

This may be due to ARM emulation on macOS (qemu). Ensure correct image base and dependencies are installed.

---

## 7. Environment Variables

Fill in your `.env` file with the following:

```env
COUCHDB_USERNAME=admin
COUCHDB_PASSWORD=password
COUCHDB_DBNAME=chiller
COUCHDB_URL=http://couchdb:5984/

WATSONX_APIKEY=
WATSONX_PROJECT_ID=
WATSONX_URL=

PATH_TO_DATASETS_DIR=/opt/conda/envs/assetopsbench/lib/python3.12/site-packages/tsfmagent/data/datasets
PATH_TO_MODELS_DIR=/opt/conda/envs/assetopsbench/lib/python3.12/site-packages/tsfmagent/data/tsfm_models
PATH_TO_OUTPUTS_DIR=/opt/conda/envs/assetopsbench/lib/python3.12/site-packages/tsfmagent/output

SKYSPARK_USERNAME=
SKYSPARK_PASSWORD=
SKYSPARK_URL=

OPENAI_API_KEY=
```

### Notes
- **Skyspark credentials**: Required if integrating with building management systems.
- **OpenAI API key**: Required if using LLM-based agents.

---

## 8. References

- [AssetOpsBench GitHub](https://github.com/IBM/AssetOpsBench)
- [Benchmark README](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/README.md)
- [Rancher Desktop](https://rancherdesktop.io/)
- [Docker Hub](https://hub.docker.com/)
- [Quay.io](https://quay.io/) (Red Hat container registry)


## Benchmark in Docker Environment

## Create your own environment




### Pre-built Docker Images
We provide two base Docker images `quay.io/assetopsbench/assetopsbench-basic` and `quay.io/assetopsbench/assetopsbench-extra` which serves as the AssetOpsBench environment. The base images provide
- `quay.io/assetopsbench/assetopsbench-basic`: A minimal Python 3.12 environment with only essential Python packages installed. Suitable for lightweight benchmarking tasks or custom setup.
- `quay.io/assetopsbench/assetopsbench-extra`: A full-featured Python 3.12 environment with all components pre-installed, including Individual agents (e.g., IoT Agent), MetaAgent, and AgentHive.

The python environment in both docker images use Conda to manage. The name of pre-built Conda environment is `assetopsbench`. For full list of installed packages, please refer to [basic_requirements.txt](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/basic_requirements.txt) and [extra_requirements.txt](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/extra_requirements.txt).

### CouchDB Setup

To streamline access to asset data during benchmarking, we use a Dockerized CouchDB instance. This setup allows for easy loading, querying, and management of benchmark datasets within a consistent environment.

A sample [docker-compose.yml](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/docker-compose.yml) is provided to help you start CouchDB locally with default configuration and credentials. We use [couchdb_setup.sh](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/couchdb_setup.sh) as starting command to create a CouchDB database populating with a [sample Chiller dataset](https://github.com/IBM/AssetOpsBench/blob/main/src/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv).


### Run Benchmark Script
In [docker-compose.yml](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/docker-compose.yml), we mount the benchmark scripts ([entrypoint.sh](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/entrypoint.sh) and [run_plan_execute.py](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/run_plan_execute.py) and scenario files as volumes to assetopsbench container.

Now we run the following command,

```commandline
cd /path/to/AssetOpsBench
chmod +x benchmark/entrypoint.sh
docker-compose -f benchmark/docker-compose.yml build
docker-compose -f benchmark/docker-compose.yml up
```

You can use [.env](https://github.com/IBM/AssetOpsBench/blob/main/benchmark/.env) or add your environment variables to `.env` which can be passed to containers through `docker-compose.yml`.

 Note that in `entrypoint.sh`, we activate the python environment by `conda activate assetopsbench` first.
 
### Customization

#### Customize Your Python Environment
You can install additional packages (e.g. your own agents) by using `quay.io/assetopsbench/assetopsbench-basic` or `quay.io/assetopsbench/assetopsbench-extra` as base image. To build your own image, here is an example Dockerfile,
```commandline
FROM quay.io/assetopsbench/assetopsbench-basic

ARG CONDA_ENV_NAME=assetopsbench
COPY your_requirements.txt /tmp/your_requirements.txt
RUN opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install -r /tmp/your_requirements.txt
```

#### Customize Your Run Script
Use your own run script and update `docker-compose.yml`. Recommend mount your scripts as volumes for ease development.
