import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# AWS Credentials and Region
AWS_ACCESS_KEY_ID =
AWS_SECRET_ACCESS_KEY =
AWS_REGION_NAME =


# Initialize AWS clients
def initialize_clients():
    try:
        clients = {
            'rekognition_client': boto3.client('rekognition', aws_access_key_id=AWS_ACCESS_KEY_ID,
                                               aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                               region_name=AWS_REGION_NAME),

        }
        print("AWS clients initialized successfully.")
        return clients
    except NoCredentialsError as e:
        print(f"AWS credentials not found or incorrect: {e}")
        return None
    except ClientError as e:
        print(f"Error initializing AWS clients: {e}")
        return None


def rekognition_client():
    try:
        return boto3.client('rekognition', aws_access_key_id=AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                            region_name=AWS_REGION_NAME)
    except NoCredentialsError as e:
        print(f"AWS credentials not found or incorrect: {e}")
        return None
    except ClientError as e:
        print(f"Error initializing AWS Rekognition client: {e}")
        return None


# Initialize AWS clients
aws_clients = initialize_clients()
rekognition = rekognition_client()
