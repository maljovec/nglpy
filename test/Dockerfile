FROM ubuntu:latest
LABEL version="0.0"
LABEL description="This is a test Ubuntu build for trying to install nglpy."
RUN apt-get -y update && apt-get install -y python-pip
RUN pip install --upgrade pip
RUN pip install --index-url https://test.pypi.org/simple/ nglpy
