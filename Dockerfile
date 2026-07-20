FROM python:3.12.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY schemas ./schemas

RUN pip install --no-cache-dir . \
	&& useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["aep", "serve", "--host", "0.0.0.0", "--port", "8000"]
