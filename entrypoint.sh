#!/bin/bash
python manage.py collectstatic --no-input
python manage.py makemigrations glacier_monitoring_app
python manage.py migrate
exec "$@"