FROM ubuntu:latest
RUN apt-get update && apt-get install -y 
RUN apt-get install -y python3 python3-pip
RUN apt-get clean
WORKDIR /home/app
COPY . .
RUN python3 -m pip install streamlit pandas xgboost streamlit_lottie openpyxl --break-system-packages
EXPOSE 8501
CMD ["streamlit", "run", "frontend.py"]
