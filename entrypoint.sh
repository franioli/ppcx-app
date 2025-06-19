#!/bin/bash
python ./planpincieux/amanage.py collectstatic --no-input
python ./planpincieux/amanage.py makemigrations glacier_monitoring_app
python ./planpincieux/amanage.py migrate
exec "$@"