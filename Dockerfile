# syntax=docker/dockerfile:1.7
# Two images from one file: `api` (Spark and LightGBM, so it needs a Java 17 runtime) and `ui`
# (Streamlit only, no model code). Build one with --target api or --target ui.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.12.6 AS uv

# ---- locked dependency install for the API -------------------------------------------------
FROM ${PYTHON_IMAGE} AS api-build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_EDITABLE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-group ui --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-group ui \
    # dependencies ship their own test suites; nothing imports them at runtime
    && find /app/.venv -type d -name tests -prune -exec rm -rf {} + \
    && find /app/.venv -type d -name __pycache__ -prune -exec rm -rf {} +

# ---- locked dependency install for the UI ---------------------------------------------------
FROM ${PYTHON_IMAGE} AS ui-build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group ui --no-install-project

# ---- minimal Java 17 runtime for Spark (jlink), instead of the full Debian JRE ---------------
FROM eclipse-temurin:17-jdk-jammy AS jre
RUN jlink --strip-debug --no-man-pages --no-header-files --compress=2 \
    --add-modules java.base,java.desktop,java.scripting,java.compiler,java.logging,java.sql,java.management,java.naming,java.xml,\
java.security.jgss,java.security.sasl,java.instrument,java.rmi,java.prefs,java.net.http,\
jdk.unsupported,jdk.incubator.vector,jdk.net,jdk.crypto.ec,jdk.management,jdk.zipfs,jdk.security.auth,jdk.naming.dns \
    --output /opt/jre

# ---- API runtime ----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS api
# libgomp1 is the OpenMP runtime LightGBM loads; the slim Python image does not have it
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
COPY --from=jre /opt/jre /opt/java
ENV JAVA_HOME=/opt/java \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=api-build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app config ./config
COPY --chown=app:app scripts/capture_darwin.py scripts/make_fixture_store.py ./scripts/
RUN mkdir -p data mlruns && chown app:app data mlruns
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["rail-risk", "serve", "--host", "0.0.0.0", "--port", "8000"]

# ---- UI runtime -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS ui
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOME=/home/app \
    RAIL_API_URL=http://api:8000
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=ui-build --chown=app:app /app/.venv /app/.venv
# only the page and its helpers: no model, Spark or data code is in this image
COPY --chown=app:app src/rail_risk/__init__.py ./src/rail_risk/__init__.py
COPY --chown=app:app src/rail_risk/ui ./src/rail_risk/ui
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "src/rail_risk/ui/app.py", "--server.address", "0.0.0.0", \
     "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false", \
     "--client.toolbarMode", "minimal"]
