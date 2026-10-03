FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AW_DATA_DIR=/data
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY workbench ./workbench
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-deps .
RUN useradd --create-home --uid 10001 agent && mkdir /data && chown agent:agent /data
USER agent
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["python", "-m", "workbench", "--host", "0.0.0.0", "--port", "8080"]
