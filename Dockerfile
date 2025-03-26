FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY huobi ./huobi
RUN pip install keras numpy pandas matplotlib yfinance scikit-learn seaborn
RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk
#COPY chart ./chart
ENV PYTHONPATH=/app
ENV DISPLAY=172.17.0.1:0

ENTRYPOINT ["python", "/app/chart/yf_keras.py"]
#ENTRYPOINT ["bash"]