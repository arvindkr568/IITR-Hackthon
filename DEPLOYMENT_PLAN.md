# 🚀 AWS Cloud Deployment Strategy & Production Plan
**Project:** Coupon Acceptance ML Hackathon  
**Target Environment:** AWS (Cloud Hosting) + GitHub CI/CD + Streamlit Inference Application  
**Version:** 1.0 | September 2026

---

## 1. Executive Summary & Architecture Overview

The deployment architecture is designed as a **thin, production-style inference layer**. Model training and validation are performed in the local development environment (`src/train_final.py`). The serialized winning pipeline (`models/model_pipeline.joblib`) is packaged into a containerized Streamlit application hosted on AWS.

```
+------------------+        +--------------------+        +-----------------------+
|  Developer /     |  git   | GitHub Repository  | CI/CD  |  GitHub Actions       |
|  Cursor IDE      | ----=> | (Main Branch)      | ----=> |  Pytest & Build Pass  |
+------------------+        +--------------------+        +-----------------------+
                                                                      |
                                                                      v
                                                          +-----------------------+
                                                          |  AWS ECR / Docker Hub |
                                                          +-----------------------+
                                                                      |
                                                                      v
+------------------+        +--------------------+        +-----------------------+
|  End User /      | HTTP   | Live AWS Streamlit | Loads  | Persisted ML Pipeline |
|  Hackathon Judge | <====> | App (Port 8501)    | <====> | (model_pipeline.joblib)|
+------------------+        +--------------------+        +-----------------------+
```

---

## 2. Recommended Deployment Options on AWS

### Option A: AWS App Runner (Recommended for Speed & Reliability)
AWS App Runner provides fully managed container deployment with built-in automatic HTTPS, scaling, and health monitoring.

1. **Build Container Image & Push to Amazon ECR:**
   ```bash
   # Log in to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

   # Create repository
   aws ecr create-repository --repository-name coupon-acceptance-app

   # Tag and push image
   docker build -t coupon-acceptance-app .
   docker tag coupon-acceptance-app:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/coupon-acceptance-app:latest
   docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/coupon-acceptance-app:latest
   ```

2. **Create App Runner Service:**
   - Source: Amazon ECR repository
   - Port: `8501`
   - CPU / Memory: `1 vCPU / 2 GB RAM`
   - Health Check Path: `/_stcore/health`
   - Resulting Public URL: `https://<random-id>.us-east-1.awsapprunner.com`

---

### Option B: AWS EC2 (t3.medium Ubuntu Instance + Nginx)

For standard VM deployment:

1. **Launch Instance:**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance Type: `t3.medium` (2 vCPU, 4 GB RAM)
   - Security Group Rules: Allow `SSH (22)`, `HTTP (80)`, `HTTPS (443)`, `Custom TCP (8501)`.

2. **Provision EC2 Environment:**
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git nginx
   git clone https://github.com/<your-username>/coupon-acceptance-ml.git
   cd coupon-acceptance-ml
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Systemd Background Service (`/etc/systemd/system/streamlit.service`):**
   ```ini
   [Unit]
   Description=Streamlit Coupon Acceptance App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/coupon-acceptance-ml
   ExecStart=/home/ubuntu/coupon-acceptance-ml/venv/bin/streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start Service & Verify:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit
   sudo systemctl start streamlit
   sudo systemctl status streamlit
   ```

---

## 3. Environment Variables & Security Constraints

- **No Secrets in Source Control:** All AWS credentials, access keys, or API tokens must be passed via environment variables or AWS Systems Manager Parameter Store.
- **Inbound Access Control:** Inbound SSH (port 22) restricted strictly to administrator IP addresses. Public access allowed only on ports 80/443 or 8501.

---

## 4. Health Verification Procedure

1. **Automated Health Check Endpoint:**
   `GET http://<AWS_HOST>:8501/_stcore/health` -> Expect HTTP Status `200 OK`.
2. **Inference Verification Test:**
   Access live application URL, submit customer scenario, verify acceptance probability score rendering within 500ms.

---

## 5. MLOps & Future Cloud Scope

- **Automated Retraining Trigger:** AWS Lambda triggered by new batch customer data in S3.
- **Model Registry & Monitoring:** AWS SageMaker Model Registry + Evidently AI for distribution drift monitoring.
- **Microservices API:** Packaging pipeline behind AWS API Gateway + AWS Lambda / FastAPI for e-commerce system integration.
