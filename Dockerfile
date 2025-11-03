# # Use a smaller base image that includes gcc
# FROM python:3.10-slim-buster

# WORKDIR /app

# # Copy requirements and setup first
# COPY requirements.txt setup.py ./

# # Install build dependencies, install packages, then remove build deps
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends gcc python3-dev \
#     && pip install --no-cache-dir -r requirements.txt \
#     && pip install -e . \
#     && apt-get remove -y gcc python3-dev \
#     && apt-get autoremove -y \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Copy the rest of the code
# COPY . .

# # Create directories and set permissions
# RUN mkdir -p /app/frontend /app/outputs /app/uploads /app/logs /tmp/matplotlib \
#     && chmod -R 777 /app/outputs /app/uploads /app/logs /tmp/matplotlib \
#     && chown -R nobody:nogroup /app /tmp/matplotlib

# # Environment setup
# ENV PYTHONPATH="/app:${PYTHONPATH:-}" \
#     MPLCONFIGDIR="/tmp/matplotlib"

# # Switch to non-root user
# USER nobody

# EXPOSE 8000

# # Run with package-based import
# CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-slim-buster

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create required directories and set permissions
RUN mkdir -p /app/frontend /app/outputs /app/uploads /app/logs /tmp/matplotlib \
    && chmod -R 777 /app/outputs /app/uploads /app/logs /tmp/matplotlib

# Set Matplotlib config dir to writable location
ENV MPLCONFIGDIR=/tmp/matplotlib

# Expose port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]