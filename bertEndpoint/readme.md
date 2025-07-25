# Quick Setup Process ⚙️
You need Windows WSL, Docker and GoogleCloudCLI to be installed. :p🪦

1. [https://learn.microsoft.com/en-us/windows/wsl/install](https://learn.microsoft.com/en-us/windows/wsl/install)
2. [https://cloud.google.com/sdk/docs/install](https://docs.docker.com/desktop/setup/install/windows-install/)
3. [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

To start local prediction you can use *model/localInference.py*
 

To start fastapi locally
```
uvicorn main:app --port 8080
```

To build the DOCKER file on windwos go to
```
#Start WSL
type cd /mnt/c/Users/User/Desktop/bertEndpoint/
then type docker build -t ellabert .
then type docker run -p 8080:8080 ellabert
```

To push to GCP artifact registry
```
gcloud auth configure-docker asia-southeast2-docker.pkg.dev
gcloud artifacts repositories create ellabert-docker-repo --repository-format=docker --location=asia-southeast2 --description="A Self Hosted Bert Classification called Ella FatAPI backend"
docker tag aa70b2a9af44 asia-southeast2-docker.pkg.dev/gcp-chatbot-454708/ellabert-docker-repo/ellabert-image
docker push asia-southeast2-docker.pkg.dev/gcp-chatbot-454708/ellabert-docker-repo/ellabert-image
```
Made by 🦍 idac-warot 07-2025
 
