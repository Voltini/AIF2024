# Use an official Python runtime as the parent image
FROM python:3.8-slim
COPY requirements.txt requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run mnist_api.py when the container launches
CMD ["python", "mnist_api.py", "--model_path", "weights/mnist_net.pth"]
