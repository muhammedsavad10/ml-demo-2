import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time
import boto3

role="arn:aws:iam::477010600979:role/service-role/AmazonSageMaker-ExecutionRole-20260511T140775"

boto_session=boto3.Session(region_name="eu-north-1")
session=sagemaker.Session(boto_session=boto_session)

endpoint_name=f"iris-endpoint"--{int(time.time())}

model=SKlearnModel(
    model_data="s3://my-ml-model-buckets/model.tar.gz",
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1",
    sagemaker_session=session
)

sm_client=boto3.client("sagemaker",region_name="eu-north-1")

predictor=model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print("Deployment complete")