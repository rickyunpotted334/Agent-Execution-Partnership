# Deployment

Local container deployment:

```powershell
docker build -t aep:local .
docker run -p 8000:8000 aep:local
```

Compose deployment:

```powershell
docker compose up --build
```
