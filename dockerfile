FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirement.txt

CMD ['uvicorn','app:app','--host','0.0.0.0','--port','8000']
