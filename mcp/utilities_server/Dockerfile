FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY tests/test_*.py tests/

# Expose the port for SSE transport
EXPOSE 8001

CMD ["fastmcp", "run", "main.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8001"]
