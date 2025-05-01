 sudo rm -rf chroma_data

 sudo rm -rf uploads_index.json

 cd Docker

 cd docker build -t ai-file-manager:latest .

 sudo docker-compose up -d