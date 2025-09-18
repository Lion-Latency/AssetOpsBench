# EAM Lite

EAM Lite is an EAM solution inspired database platform.
It contains data, api server and mcp tools that can be used during agent development.

It is the backbone of the AssetOpsBench's data layer.

## Build

from the top level fodler of the project run

```sh
podman build -t eamlite-postgres:latest -f db/Containerfile .
podman build -t eamlite-api -f src/Containerfile .
```

## Run

### Manual run

```sh
podman network create eamlite
podman run --name eampg -e POSTGRES_USER=eamlite -e POSTGRES_PASSWORD=eamlite -e POSTGRES_DB=eamlite -p 5432:5432 --network eamlite -d eamlite-postgres
sleep 5
podman run -e DATABASE_URL=postgresql+psycopg://eamlite:eamlite@eampg:5432/eamlite --network eamlite -p 8000:8000 -d eamlite-api
```

### Compose

create a `.env` file in the root of the project, like

```sh
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
```

then run `podman compose up`

in both cases the api will be available at `localhost:8000` and the doc at `localhost:8000/doc`
