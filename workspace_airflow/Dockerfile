FROM apache/airflow:2.10.1

USER root

# Update package list and install Git
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

USER airflow

RUN pip install torch torchvision matplotlib