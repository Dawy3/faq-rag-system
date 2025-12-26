# Use an official python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed package specified in requirements.txt
RUN pip install --no-cash-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Run app.main:app when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]