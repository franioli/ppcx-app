FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cifs-utils \
    gdal-bin \
    libgdal-dev \
    postgresql-client \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GDAL_LIBRARY_PATH=/usr/lib/libgdal.so

# Set work directory
WORKDIR /ppcx

# Install dependencies using uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy project files and   Set the working directory to the Django app
COPY ./app ./app
WORKDIR /ppcx/app

EXPOSE 8000

# Start Django server
CMD ["uv", "run", "--project", "/ppcx", "python", "manage.py", "runserver", "0.0.0.0:8000"]