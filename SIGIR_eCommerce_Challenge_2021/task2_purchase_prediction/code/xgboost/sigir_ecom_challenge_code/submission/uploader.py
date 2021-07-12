"""

    Sample script showing how to submit a prediction json file to the AWS bucket assigned to you for the challenge.
    Credentials are in your sign-up e-mail: please refer to the full project README for the exact format of the file
    and the naming convention you need to respect.

    Make sure to duplicate the .env.local file as an .env file in this folder, and fill it with the right values
    (or alternatively, set up the corresponding env variables).

    Required packages can be found in the requirements.txt file in this folder.

"""

import os
from datetime import datetime

import boto3
from dotenv import load_dotenv

# load envs from env file
load_dotenv(
    verbose=True,
    dotenv_path="sigir_ecom_challenge_code/submission/upload.env",
)

# env info should be in your env file
BUCKET_NAME = os.getenv("BUCKET_NAME")  # you received it in your e-mail
EMAIL = os.getenv("EMAIL")  # the e-mail you used to sign up
PARTICIPANT_ID = os.getenv("PARTICIPANT_ID")  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")  # you received it in your e-mail


def upload_submission(local_file: str, task: str):
    """
    Thanks to Alex Egg for catching the bug!

    :param local_file: local path, may be only the file name or a full path
    :param task: rec or cart
    :return:
    """

    print("Starting submission at {}...\n".format(datetime.utcnow()))
    # instantiate boto3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name="us-west-2",
    )
    s3_file_name = os.path.basename(local_file)
    # prepare s3 path according to the spec
    s3_file_path = "{}/{}/{}".format(
        task, PARTICIPANT_ID, s3_file_name
    )  # it needs to be like e.g. "rec/id/*.json"
    # upload file
    s3_client.upload_file(local_file, BUCKET_NAME, s3_file_path)
    # say bye
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))

    return


if __name__ == "__main__":
    # LOCAL_FILE needs to be a json file with the format email_epoch time in ms - email should replace @ with _
    LOCAL_FILE = "{}_1616887274000.json".format(EMAIL.replace("@", "_"))
    TASK = "rec"  # 'rec' or 'cart'
    upload_submission(local_file=LOCAL_FILE, task=TASK)
