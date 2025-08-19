FROM python:3.11-slim

# Copy project into image
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["bash"]