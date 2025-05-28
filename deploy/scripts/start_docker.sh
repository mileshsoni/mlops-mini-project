#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 879381240041.dkr.ecr.eu-north-1.amazonaws.com
docker pull 879381240041.dkr.ecr.eu-north-1.amazonaws.com/milesh_ecr:v5
docker stop my-container || true
docker rm my-container || true
docker run -d -p 80:5000 -e DAGSHUB_PAT=d679807467b2274934bf800bc70fdaf185952270  --name my-app 879381240041.dkr.ecr.eu-north-1.amazonaws.com/milesh_ecr:v5


