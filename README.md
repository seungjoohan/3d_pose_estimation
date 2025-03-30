## 3D Pose Estimation App

To build Docker image and test locally
`docker build -t pose-estimation-app .`
`docker run -p 8080:8080 -e PORT=8080 -e GOOGLE_CLOUD_PROJECT=pose-app" -e FLASK_ENV=production pose-estimation-app`

To build Docker image and Push to Cloud Run
`gcloud builds submit --config=cloudbuild.yaml`