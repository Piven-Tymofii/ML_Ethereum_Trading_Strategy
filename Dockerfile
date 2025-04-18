# Use a lightweight Python image
FROM python:3.11-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

# Copy the dependencies file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Command to run the bot
CMD ["python", "tg_bot_signals.py"]