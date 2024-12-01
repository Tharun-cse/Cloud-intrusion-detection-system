# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000
ENV NAME OpentoAll
CMD ["python","app.py"]

