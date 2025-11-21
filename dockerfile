FROM python:3.9-slim 
WORKDIR /app
# Copier les fichiers
COPY . .
# Installer les dépendances directement
RUN pip install fastapi==0.104.1 uvicorn==0.24.0 tensorflow==2.13.0 pillow==10.0.1 python-multipart==0.0.6
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]