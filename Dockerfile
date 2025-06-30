FROM python:3.10-slim

# Install GDAL and other dependencies
RUN apt-get update && apt-get install -y \
    cifs-utils \
    gdal-bin \
    libgdal-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GDAL_LIBRARY_PATH=/usr/lib/libgdal.so

# Set work directory
WORKDIR /ppcx

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy project files
COPY ./app ./app

# Copy entrypoint script
# COPY entrypoint.sh ./
# RUN chmod +x ./entrypoint.sh

# Set the working directory to the Django app
WORKDIR /ppcx/app

EXPOSE 8000

# Start Django server
CMD ["uv", "run", "--project", "/ppcx", "python", "manage.py", "runserver", "0.0.0.0:8000"]