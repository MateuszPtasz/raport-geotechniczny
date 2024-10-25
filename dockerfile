
FROM python:3.9-slim


RUN apt-get update && apt-get install -y \
    libpango1.0-0 \
    libglib2.0-0 \
    libxml2 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .


RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY . .


EXPOSE 8000


CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000"]
