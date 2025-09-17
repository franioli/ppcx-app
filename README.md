# PPCX App

## Setup and Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your specific configuration:

```bash
# Database connection settings
DB_NAME=your_database_name
DB_USER=your_db_username
DB_PASSWORD=your_db_password
DB_HOST=your_db_host_ip
DB_PORT=5432

# Django settings
DJANGO_SETTINGS_MODULE=planpincieux.settings
SECRET_KEY="your-secret-key-here"
DEBUG=True # Disable in production
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0,your_server_ip

# SMB/CIFS credentials for network storage
SMB_USERNAME=your_smb_username
SMB_PASSWORD=your_smb_password
```

## SMB/CIFS Volume Setup

Before starting the application, you must manually create the SMB/CIFS Docker volume using your credentials from the `.env` file. This ensures reliable mounting.
```bash
export SMB_USERNAME=$(grep SMB_USERNAME .env | cut -d '=' -f2)
export SMB_PASSWORD=$(grep SMB_PASSWORD .env | cut -d '=' -f2)
docker volume create \
  --driver local \
  --opt type=cifs \
  --opt device=//150.145.51.244/GMG_ftp2/FMS \
  --opt o=username=$SMB_USERNAME,password=$SMB_PASSWORD,vers=3.0,file_mode=0777,dir_mode=0777,uid=1000,gid=1000,iocharset=utf8 \
  --name fms_data
```

- Make sure `cifs-utils` is installed on your host.
- The volume will be available as `/data` inside the container.

Check that the volume is created successfully:

```bash
 docker run --rm -it \
  --mount source=fms_data,target=/mnt/fms \
  alpine:latest \
  sh -c "ls -la /mnt/fms"
```

## 3. Start the application

```bash
docker-compose up -d
```


# For developers

List of useful commands to work with the docker container

- Build and run the container in detached mode:
```bash
docker compose up -d
```

- Force rebuild docker image:
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

# Populate the database with images and DIC

To populate the database with images and DIC data, you can use the scripts provided in the `app` directory.
These scripts can be run inside the docker container using `docker compose exec` or from outside the container if you have the necessary dependencies installed and you have access to the database.