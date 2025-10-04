# PPCX App

Containerized Django + PostGIS stack with two main services:
- `db`: PostgreSQL with PostGIS extension for spatial data support.
- `web`: Django application server using Django.

## Setup and Configuration


### 1. Set up secrets (no .env file anymore)

This stack uses Docker secrets, mounted as files in containers under /run/secrets. Secrets are not exposed as environment variables.

Required secrets (create once on the host):
- ~/secrets/db_password
- ~/secrets/django_secret_key

Create and secure the secrets:.

```bash
mkdir -p ~/secrets
chmod 700 ~/secrets

printf 'your-db-password\n' > ~/secrets/db_password
printf 'your-django-secret-key\n' > ~/secrets/django_secret_key
chmod 600 ~/secrets/*
```

Notes:
- docker-compose.yaml references these files as secrets; they are mounted read-only into the web and db containers.
- If ~ is not expanded on your system, replace it with the absolute path (/home/fioli/secrets) in docker-compose.yaml.

### 2. Update docker-compose.yaml

Edit `docker-compose.yaml` to ensure the paths to your secrets are correct. Replace `~/secrets/db_password` and `~/secrets/django_secret_key` with the absolute paths if necessary.

Update volume paths if you want to change where data is stored on the host.

PostgreSQL data is persisted in a Docker named volume db_data. Recreating the stack will not delete data unless you explicitly remove the volume: 
  - Volume name (actual name will be prefixed by the Compose project, e.g., ppcx-app_db_data)
  - Mount path in the db container: /var/lib/postgresql/data

### 3. Start the stack

Build and run:
```bash
docker compose up -d
```

Run migrations:
```bash
docker compose exec web uv run python manage.py migrate
```

Access the app:
- http://localhost:8080 (or `http://<your-server-ip>:8080`)

Connect to Postgres from the host (optional):
```bash
psql "postgresql://postgres:$(cat ~/secrets/db_password)@localhost:5434/planpincieux"
```

## Developer quick reference

List of useful commands to work with the docker container

- Rebuild images without cache:
```bash
docker compose build --no-cache
```

- Connect shell to container:
```bash
docker compose exec web /bin/bash
```

- Install new dependencies and restart the container:

```bash
docker compose exec web uv pip install <package_name>
docker compose restart web
```

```bash
docker compose exec web uv pip install <package_name>
docker compose restart web
```

- Make migration inside docker container: 

```bash
docker compose exec web uv run ./manage.py makemigrations ppcx_app
docker compose exec web uv run ./manage.py migrate
```

## Populate the database with images and DIC

To populate the database with images and DIC data, you can use the scripts provided in the `app` directory.
These scripts can be run inside the docker container using `docker compose exec` or from outside the container if you have the necessary dependencies installed and you have access to the database.



## Rotating secrets

1) Update the file under ~/secrets (e.g., echo 'newpass' > ~/secrets/db_password).
2) Redeploy:
```bash
docker compose up -d
```
The container will read the updated secret from /run/secrets on restart.