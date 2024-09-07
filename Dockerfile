FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that Flask will run on
EXPOSE 5000

# Update the CMD to bind to all interfaces
CMD ["flask", "run", "--host", "0.0.0.0"]