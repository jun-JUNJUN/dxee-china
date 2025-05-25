FROM python:3.10-slim

WORKDIR /app

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Clone and install meilisearch-python-sdk
#RUN git clone https://github.com/sanders41/meilisearch-python-sdk.git /tmp/meilisearch-sdk && \
#    cd /tmp/meilisearch-sdk && \
#    git checkout v4.6.0 && \
#   pip install -e . && \
#   rm -rf /tmp/meilisearch-sdk

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "apps:create_app()"] 
