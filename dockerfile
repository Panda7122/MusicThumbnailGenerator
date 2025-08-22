# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
RUN mkdir -p /app/kaggleData
RUN mkdir -p /app/savePoint
COPY ./requirements.txt /app
COPY ./lyrisOnlyModel.py /app
COPY ./kaggleData/small_song_lyrics.csv  /app/kaggleData

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install CLIP from GitHub
# RUN pip3 install git+https://github.com/openai/CLIP.git

# Make port 7122 available to the world outside this container
EXPOSE 7122

# Define environment variable
ENV MODEL_ENV production

CMD ["python3", ,"lyrisOnlyModel.py"]