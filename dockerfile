FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirment.txt
COPY . .
CMD ['uvicorn','app:app','--host','0.0.0.0','--port','8000']
