# Python 3.14 и локально, и в проде: проверено, что pandas 3.0 и numpy 2.5
# ставятся колёсами без сборки, поэтому расходиться версиями незачем.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CRYPTOMCP_TRANSPORT=streamable-http \
    CRYPTOMCP_HOST=0.0.0.0 \
    CRYPTOMCP_PORT=8000 \
    CRYPTOMCP_CONFIG=/app/config.yaml \
    CRYPTOMCP_JOURNAL=/app/journal/squeeze.jsonl

WORKDIR /app

# Зависимости отдельным слоем: правка кода не приводит к переустановке pandas.
COPY pyproject.toml README.md ./
COPY cryptomcp/__init__.py ./cryptomcp/
RUN pip install --no-cache-dir "mcp>=2.1,<3" "httpx>=0.28" "pandas>=2.2" \
        "numpy>=1.26" "pyyaml>=6.0" "uvicorn>=0.30"

COPY cryptomcp/ ./cryptomcp/
COPY config.example.yaml ./config.yaml

# Непривилегированный пользователь: контейнер не хранит секретов, но и root
# внутри ему не нужен.
RUN useradd --create-home --uid 10001 cryptomcp \
    && mkdir -p /app/journal \
    && chown -R cryptomcp:cryptomcp /app
USER cryptomcp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "cryptomcp.server"]
