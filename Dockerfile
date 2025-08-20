FROM python:3.11-slim
WORKDIR /app
COPY ci-requirements.txt /app/
RUN pip install --no-cache-dir -r ci-requirements.txt
COPY . /app
CMD ["python","lab_v10/scripts/param_count.py"]