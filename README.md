## 3D Pose Estimation App

This is a project to showcase pose estimation model that I've been working on at ViFive. Aside from the healthcare web app ViFive provides, I'd like to demonstrate what we are able to achieve! 

You can access the app at (3d-pose-estimation)[https://pose-estimation-app-974323386375.us-central1.run.app/]. It's built using Google Cloud environments using free resources. Thus, it could take some time to load up and estimate poses (especially for a video).

To build Docker image and test locally
`docker build -t pose-estimation-app .`
`docker run -p 8080:8080 -e PORT=8080 -e GOOGLE_CLOUD_PROJECT="pose-app" -e FLASK_ENV=production pose-estimation-app`

To build Docker image and Push to Cloud Run
`gcloud builds submit --config=cloudbuild.yaml`