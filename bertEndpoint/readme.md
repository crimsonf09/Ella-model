`You nedd Docker, Windows WSL and CloudCLI to be installed. :p


https://cloud.google.com/sdk/docs/install

To start local prediction you can use
model/localInference.py
 

To start fastapi locally you would call
uvicorn main:app --port 8080

To build the DOCKER file on windwos go to
Start WSL
type cd /mnt/c/Users/User/Desktop/bertEndpoint/
then type docker build -t ellabert .
then type docker run -p 8080:8080 ellabert

To push to GCP artifact registry
gcloud auth configure-docker asia-southeast2-docker.pkg.dev
docker tag aa70b2a9af44 asia-southeast2-docker.pkg.dev/gcp-chatbot-454708/ellabert-docker-repo/ellabert-image
docker push asia-southeast2-docker.pkg.dev/gcp-chatbot-454708/ellabert-docker-repo/ellabert-image

gcloud artifacts repositories create ellabert-docker-repo --repository-format=docker --location=asia-southeast2 --description="A Self Hosted Bert Classification called Ella FatAPI backend"
gcloud builds submit --region=asia-southeast2 --tag=asia-southeast2-docker.pkg.dev/gcp-chatbot-454708/ellabert-docker-repo:tag1
