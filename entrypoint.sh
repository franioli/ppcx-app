#!/bin/bash
python ./planpincieux/manage.py collectstatic --no-input
python ./planpincieux/manage.py makemigrations glacier_monitoring_app
python ./planpincieux/manage.py migrate
exec "$@"