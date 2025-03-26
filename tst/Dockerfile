FROM python:3.10.10-alpine as backend_image
RUN apk add gcc musl-dev
RUN pip install --upgrade pip
RUN pip install APScheduler==3.6.3 requests==2.20.1 aiohttp==3.6.2 websocket_client==0.57.0 cryptography gunicorn dash==2.17.0 pandas plotly sqlalchemy