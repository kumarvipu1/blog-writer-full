# Use the correct platform (x86_64) for GCP environments
FROM --platform=linux/amd64 python:3.10.17-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/usr/local/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        wget \
        xvfb \
        xauth \
        libcairo2 \
        libpango1.0-0 \
        libgdk-pixbuf2.0-0 \
        libffi-dev \
        libjpeg-dev \
        libxml2-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libopenjp2-7 \
        libtiff5-dev \
        libwebp-dev \
        libgtk2.0-0 \
        libxtst6 \
        libxss1 \
        libgconf-2-4 \
        libnss3 \
        libasound2 \
        dos2unix \
        fonts-dejavu-core \ 
        fonts-liberation \
    && pip install --no-cache-dir -r requirements.txt \
    # Force reinstall streamlit to fix exec issues
    && pip install --no-cache-dir --force-reinstall streamlit \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 



# Copy the current directory contents into the working directory
COPY . .

# Make port 8080 available
EXPOSE 8080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8080/_stcore/health

# Start Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
