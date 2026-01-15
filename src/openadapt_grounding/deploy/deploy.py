"""Deployment module for OmniParser on AWS EC2.

Adapted from https://github.com/OpenAdaptAI/OpenAdapt/pull/943

Prerequisites:
    1. AWS credentials configured (via environment or ~/.aws/credentials)
    2. Install deploy dependencies: uv pip install openadapt-grounding[deploy]

Environment variables (or .env file):
    AWS_ACCESS_KEY_ID - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret key
    AWS_REGION - AWS region (e.g., us-east-1)
    PROJECT_NAME - Optional, defaults to "omniparser"

Usage:
    python -m openadapt_grounding.deploy start   # Deploy new instance
    python -m openadapt_grounding.deploy status  # Check status
    python -m openadapt_grounding.deploy ssh     # SSH into instance
    python -m openadapt_grounding.deploy stop    # Terminate instance
    python -m openadapt_grounding.deploy logs    # Show container logs
    python -m openadapt_grounding.deploy ps      # Show container status
    python -m openadapt_grounding.deploy build   # Build Docker image
    python -m openadapt_grounding.deploy run     # Start container
    python -m openadapt_grounding.deploy test    # Test endpoint
"""

import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
    import paramiko
except ImportError:
    raise ImportError(
        "Deploy dependencies not installed. Run: uv pip install openadapt-grounding[deploy]"
    )

from openadapt_grounding.deploy.config import settings as config


CLEANUP_ON_FAILURE = False

# Auto-shutdown infrastructure names
LAMBDA_FUNCTION_NAME = f"{config.PROJECT_NAME}-auto-shutdown"
IAM_ROLE_NAME = f"{config.PROJECT_NAME}-lambda-role"


def _get_dockerfile_path() -> Path:
    """Get path to Dockerfile in package."""
    return Path(__file__).parent / "Dockerfile"


def _get_dockerignore_path() -> Path:
    """Get path to .dockerignore in package."""
    return Path(__file__).parent / ".dockerignore"


def create_key_pair(
    key_name: str = config.AWS_EC2_KEY_NAME,
    key_path: str = config.AWS_EC2_KEY_PATH,
) -> Optional[str]:
    """Create an EC2 key pair."""
    ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)
    try:
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
        private_key = key_pair["KeyMaterial"]

        with open(key_path, "w") as key_file:
            key_file.write(private_key)
        os.chmod(key_path, 0o400)

        print(f"Key pair {key_name} created and saved to {key_path}")
        return key_name
    except ClientError as e:
        print(f"Error creating key pair: {e}")
        return None


def get_or_create_security_group_id(
    ports: list = None,
) -> Optional[str]:
    """Get existing security group or create a new one."""
    if ports is None:
        ports = [22, config.PORT]

    ec2 = boto3.client("ec2", region_name=config.AWS_REGION)

    ip_permissions = [
        {
            "IpProtocol": "tcp",
            "FromPort": port,
            "ToPort": port,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
        }
        for port in ports
    ]

    try:
        response = ec2.describe_security_groups(
            GroupNames=[config.AWS_EC2_SECURITY_GROUP]
        )
        security_group_id = response["SecurityGroups"][0]["GroupId"]
        print(f"Security group '{config.AWS_EC2_SECURITY_GROUP}' exists: {security_group_id}")

        for ip_permission in ip_permissions:
            try:
                ec2.authorize_security_group_ingress(
                    GroupId=security_group_id, IpPermissions=[ip_permission]
                )
                print(f"Added inbound rule for port {ip_permission['FromPort']}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "InvalidPermission.Duplicate":
                    pass  # Rule already exists
                else:
                    print(f"Error adding rule for port {ip_permission['FromPort']}: {e}")

        return security_group_id
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidGroup.NotFound":
            try:
                response = ec2.create_security_group(
                    GroupName=config.AWS_EC2_SECURITY_GROUP,
                    Description="Security group for OmniParser deployment",
                    TagSpecifications=[
                        {
                            "ResourceType": "security-group",
                            "Tags": [{"Key": "Name", "Value": config.PROJECT_NAME}],
                        }
                    ],
                )
                security_group_id = response["GroupId"]
                print(f"Created security group: {security_group_id}")

                ec2.authorize_security_group_ingress(
                    GroupId=security_group_id, IpPermissions=ip_permissions
                )
                print(f"Added inbound rules for ports {ports}")

                return security_group_id
            except ClientError as e:
                print(f"Error creating security group: {e}")
                return None
        else:
            print(f"Error describing security groups: {e}")
            return None


def deploy_ec2_instance(
    ami: str = config.AWS_EC2_AMI,
    instance_type: str = config.AWS_EC2_INSTANCE_TYPE,
    project_name: str = config.PROJECT_NAME,
    key_name: str = config.AWS_EC2_KEY_NAME,
    disk_size: int = config.AWS_EC2_DISK_SIZE,
) -> Tuple[Optional[str], Optional[str]]:
    """Deploy a new EC2 instance or return existing one."""
    ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
    ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)

    # Check for existing instances
    instances = ec2.instances.filter(
        Filters=[
            {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
            {"Name": "instance-state-name", "Values": ["running", "pending", "stopped"]},
        ]
    )

    existing_instance = None
    for instance in instances:
        existing_instance = instance
        if instance.state["Name"] == "running":
            print(f"Instance already running: {instance.id} @ {instance.public_ip_address}")
            break
        elif instance.state["Name"] == "stopped":
            print(f"Starting stopped instance: {instance.id}")
            ec2_client.start_instances(InstanceIds=[instance.id])
            instance.wait_until_running()
            instance.reload()
            print(f"Instance started: {instance.id} @ {instance.public_ip_address}")
            break

    if existing_instance:
        if not os.path.exists(config.AWS_EC2_KEY_PATH):
            print(f"Warning: Key file {config.AWS_EC2_KEY_PATH} not found")
            print("You need the original key to connect. Consider 'deploy stop' and restart.")
            return None, None
        return existing_instance.id, existing_instance.public_ip_address

    # Create new instance
    security_group_id = get_or_create_security_group_id()
    if not security_group_id:
        print("Failed to get security group. Aborting.")
        return None, None

    # Create new key pair
    try:
        if os.path.exists(config.AWS_EC2_KEY_PATH):
            os.remove(config.AWS_EC2_KEY_PATH)

        try:
            ec2_client.delete_key_pair(KeyName=key_name)
        except ClientError:
            pass

        if not create_key_pair(key_name):
            return None, None
    except Exception as e:
        print(f"Error managing key pair: {e}")
        return None, None

    ebs_config = {
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": disk_size,
            "VolumeType": "gp3",
            "DeleteOnTermination": True,
        },
    }

    new_instance = ec2.create_instances(
        ImageId=ami,
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        BlockDeviceMappings=[ebs_config],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": project_name}],
            },
        ],
    )[0]

    new_instance.wait_until_running()
    new_instance.reload()
    print(f"New instance created: {new_instance.id} @ {new_instance.public_ip_address}")
    return new_instance.id, new_instance.public_ip_address


def configure_ec2_instance(
    instance_id: Optional[str] = None,
    instance_ip: Optional[str] = None,
    max_ssh_retries: int = 20,
    ssh_retry_delay: int = 20,
    max_cmd_retries: int = 20,
    cmd_retry_delay: int = 30,
) -> Tuple[Optional[str], Optional[str]]:
    """Configure EC2 instance with Docker."""
    if not instance_id:
        instance_id, instance_ip = deploy_ec2_instance()

    if not instance_ip:
        return None, None

    key = paramiko.RSAKey.from_private_key_file(config.AWS_EC2_KEY_PATH)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_retries = 0
    while ssh_retries < max_ssh_retries:
        try:
            ssh_client.connect(
                hostname=instance_ip, username=config.AWS_EC2_USER, pkey=key
            )
            break
        except Exception as e:
            ssh_retries += 1
            print(f"SSH attempt {ssh_retries} failed: {e}")
            if ssh_retries < max_ssh_retries:
                print(f"Retrying in {ssh_retry_delay}s...")
                time.sleep(ssh_retry_delay)
            else:
                print("Max SSH retries reached. Aborting.")
                return None, None

    commands = [
        "sudo apt-get update",
        "sudo apt-get install -y ca-certificates curl gnupg",
        "sudo install -m 0755 -d /etc/apt/keyrings",
        "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo dd of=/etc/apt/keyrings/docker.gpg",
        "sudo chmod a+r /etc/apt/keyrings/docker.gpg",
        'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null',
        "sudo apt-get update",
        "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
        "sudo systemctl start docker",
        "sudo systemctl enable docker",
        "sudo usermod -a -G docker ${USER}",
        "sudo docker system prune -af --volumes",
        f"sudo docker rm -f {config.CONTAINER_NAME} || true",
    ]

    for command in commands:
        print(f"Executing: {command[:60]}...")
        cmd_retries = 0
        while cmd_retries < max_cmd_retries:
            stdin, stdout, stderr = ssh_client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status == 0:
                break
            else:
                error_message = stderr.read().decode()
                if "Could not get lock" in error_message:
                    cmd_retries += 1
                    print(f"dpkg locked, retrying in {cmd_retry_delay}s ({cmd_retries}/{max_cmd_retries})")
                    time.sleep(cmd_retry_delay)
                else:
                    print(f"Command failed: {error_message}")
                    break

    ssh_client.close()
    return instance_id, instance_ip


def execute_command(ssh_client: paramiko.SSHClient, command: str) -> None:
    """Execute command and stream output."""
    print(f"Executing: {command[:80]}...")
    stdin, stdout, stderr = ssh_client.exec_command(
        command, timeout=config.COMMAND_TIMEOUT
    )

    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            line = stdout.channel.recv(1024).decode("utf-8", errors="replace")
            if line.strip():
                print(line.strip())

    exit_status = stdout.channel.recv_exit_status()

    remaining = stdout.read().decode("utf-8", errors="replace")
    if remaining.strip():
        print(remaining.strip())

    if exit_status != 0:
        error = stderr.read().decode("utf-8", errors="replace")
        if error.strip():
            print(f"Error: {error.strip()}")
        raise RuntimeError(f"Command failed with status {exit_status}")


def create_auto_shutdown_infrastructure(instance_id: str) -> None:
    """Create CloudWatch Alarm and Lambda for CPU-based auto-shutdown.

    Sets up infrastructure to automatically stop the EC2 instance when
    CPU utilization drops below 5% for the configured timeout period.
    This saves costs when the instance is idle.

    Args:
        instance_id: The EC2 instance ID to monitor
    """
    lambda_client = boto3.client("lambda", region_name=config.AWS_REGION)
    iam_client = boto3.client("iam", region_name=config.AWS_REGION)
    cloudwatch_client = boto3.client("cloudwatch", region_name=config.AWS_REGION)
    sts_client = boto3.client("sts", region_name=config.AWS_REGION)

    alarm_name = f"{config.PROJECT_NAME}-CPU-Low-Alarm-{instance_id}"

    print("Setting up auto-shutdown infrastructure...")

    # Create or get IAM Role for Lambda
    role_arn = None
    try:
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        try:
            response = iam_client.create_role(
                RoleName=IAM_ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            )
            role_arn = response["Role"]["Arn"]
            print(f"Created IAM role: {IAM_ROLE_NAME}")

            # Attach required policies
            iam_client.attach_role_policy(
                RoleName=IAM_ROLE_NAME,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )
            iam_client.attach_role_policy(
                RoleName=IAM_ROLE_NAME,
                PolicyArn="arn:aws:iam::aws:policy/AmazonEC2FullAccess",
            )
            print("Attached policies to IAM role")
            print("Waiting for IAM role propagation...")
            time.sleep(15)
        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityAlreadyExists":
                response = iam_client.get_role(RoleName=IAM_ROLE_NAME)
                role_arn = response["Role"]["Arn"]
                print(f"Using existing IAM role: {IAM_ROLE_NAME}")
            else:
                raise
    except Exception as e:
        print(f"Failed to create/get IAM role: {e}")
        print("Auto-shutdown will not be configured.")
        return

    if not role_arn:
        print("Failed to obtain IAM role ARN. Skipping auto-shutdown setup.")
        return

    # Lambda function code
    lambda_code = """
import boto3
import os
import json

INSTANCE_ID = os.environ.get('INSTANCE_ID')

def lambda_handler(event, context):
    if not INSTANCE_ID:
        print("Error: INSTANCE_ID environment variable not set.")
        return {'statusCode': 500, 'body': json.dumps('Configuration error')}

    ec2 = boto3.client('ec2')
    print(f"Inactivity alarm triggered for instance: {INSTANCE_ID}")

    try:
        response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        if not response.get('Reservations') or not response['Reservations'][0].get('Instances'):
            print(f"Instance {INSTANCE_ID} not found.")
            return {'statusCode': 404, 'body': json.dumps('Instance not found')}

        state = response['Reservations'][0]['Instances'][0]['State']['Name']

        if state == 'running':
            print(f"Stopping instance {INSTANCE_ID} due to inactivity.")
            ec2.stop_instances(InstanceIds=[INSTANCE_ID])
            return {'statusCode': 200, 'body': json.dumps('Instance stop initiated')}
        else:
            print(f"Instance {INSTANCE_ID} is in state '{state}'. No action taken.")
            return {'statusCode': 200, 'body': json.dumps('Instance not running')}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}
"""

    # Create or update Lambda function
    lambda_arn = None
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("lambda_function.py", lambda_code.encode("utf-8"))
        zip_content = zip_buffer.getvalue()

        env_vars = {"Variables": {"INSTANCE_ID": instance_id}}

        try:
            func_config = lambda_client.get_function_configuration(
                FunctionName=LAMBDA_FUNCTION_NAME
            )
            lambda_arn = func_config["FunctionArn"]
            print(f"Updating existing Lambda function: {LAMBDA_FUNCTION_NAME}")

            lambda_client.update_function_code(
                FunctionName=LAMBDA_FUNCTION_NAME, ZipFile=zip_content
            )
            waiter = lambda_client.get_waiter("function_updated_v2")
            waiter.wait(
                FunctionName=LAMBDA_FUNCTION_NAME,
                WaiterConfig={"Delay": 5, "MaxAttempts": 12},
            )

            lambda_client.update_function_configuration(
                FunctionName=LAMBDA_FUNCTION_NAME,
                Role=role_arn,
                Environment=env_vars,
                Timeout=30,
                MemorySize=128,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"Creating Lambda function: {LAMBDA_FUNCTION_NAME}")
                response = lambda_client.create_function(
                    FunctionName=LAMBDA_FUNCTION_NAME,
                    Runtime="python3.9",
                    Role=role_arn,
                    Handler="lambda_function.lambda_handler",
                    Code={"ZipFile": zip_content},
                    Timeout=30,
                    MemorySize=128,
                    Description=f"Auto-shutdown for {config.PROJECT_NAME}",
                    Environment=env_vars,
                    Tags={"Project": config.PROJECT_NAME},
                )
                lambda_arn = response["FunctionArn"]

                waiter = lambda_client.get_waiter("function_active_v2")
                waiter.wait(
                    FunctionName=LAMBDA_FUNCTION_NAME,
                    WaiterConfig={"Delay": 2, "MaxAttempts": 15},
                )
            else:
                raise

        if not lambda_arn:
            raise RuntimeError("Failed to get Lambda ARN")

        # Create CloudWatch Alarm
        evaluation_periods = max(1, config.INACTIVITY_TIMEOUT_MINUTES // 5)
        threshold_cpu = 5.0

        print(f"Creating CloudWatch alarm: {alarm_name}")
        print(f"  Will stop instance if CPU < {threshold_cpu}% for {evaluation_periods * 5} minutes")

        # Delete existing alarm first
        try:
            cloudwatch_client.delete_alarms(AlarmNames=[alarm_name])
        except ClientError:
            pass

        # Get account ID for alarm ARN
        account_id = sts_client.get_caller_identity()["Account"]
        alarm_arn = f"arn:aws:cloudwatch:{config.AWS_REGION}:{account_id}:alarm:{alarm_name}"

        cloudwatch_client.put_metric_alarm(
            AlarmName=alarm_name,
            AlarmDescription=f"Stop {instance_id} if CPU < {threshold_cpu}% for {evaluation_periods * 5} mins",
            ActionsEnabled=True,
            AlarmActions=[lambda_arn],
            MetricName="CPUUtilization",
            Namespace="AWS/EC2",
            Statistic="Average",
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            Period=300,
            EvaluationPeriods=evaluation_periods,
            Threshold=threshold_cpu,
            ComparisonOperator="LessThanThreshold",
            TreatMissingData="breaching",
            Tags=[{"Key": "Project", "Value": config.PROJECT_NAME}],
        )

        # Grant CloudWatch permission to invoke Lambda
        statement_id = f"AllowCloudWatchAlarm_{alarm_name}"
        try:
            lambda_client.remove_permission(
                FunctionName=LAMBDA_FUNCTION_NAME, StatementId=statement_id
            )
        except ClientError:
            pass

        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="cloudwatch.amazonaws.com",
            SourceArn=alarm_arn,
        )

        print(f"Auto-shutdown configured: instance will stop after {config.INACTIVITY_TIMEOUT_MINUTES} minutes of low CPU")

    except Exception as e:
        print(f"Error setting up auto-shutdown: {e}")
        print("Deployment will continue but auto-shutdown may not work.")


def cleanup_auto_shutdown_infrastructure() -> None:
    """Clean up auto-shutdown Lambda, CloudWatch alarms, and IAM role."""
    lambda_client = boto3.client("lambda", region_name=config.AWS_REGION)
    cloudwatch_client = boto3.client("cloudwatch", region_name=config.AWS_REGION)
    iam_client = boto3.client("iam", region_name=config.AWS_REGION)

    # Delete CloudWatch alarms
    try:
        alarm_prefix = f"{config.PROJECT_NAME}-CPU-Low-Alarm-"
        paginator = cloudwatch_client.get_paginator("describe_alarms")
        alarms_to_delete = []
        for page in paginator.paginate(AlarmNamePrefix=alarm_prefix):
            for alarm in page.get("MetricAlarms", []):
                alarms_to_delete.append(alarm["AlarmName"])

        if alarms_to_delete:
            print(f"Deleting CloudWatch alarms: {alarms_to_delete}")
            cloudwatch_client.delete_alarms(AlarmNames=alarms_to_delete)
    except Exception as e:
        print(f"Error deleting CloudWatch alarms: {e}")

    # Delete Lambda function
    try:
        lambda_client.delete_function(FunctionName=LAMBDA_FUNCTION_NAME)
        print(f"Deleted Lambda function: {LAMBDA_FUNCTION_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            print(f"Error deleting Lambda: {e}")

    # Delete IAM role
    try:
        # Detach policies first
        attached_policies = iam_client.list_attached_role_policies(
            RoleName=IAM_ROLE_NAME
        ).get("AttachedPolicies", [])
        for policy in attached_policies:
            iam_client.detach_role_policy(
                RoleName=IAM_ROLE_NAME, PolicyArn=policy["PolicyArn"]
            )

        iam_client.delete_role(RoleName=IAM_ROLE_NAME)
        print(f"Deleted IAM role: {IAM_ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            print(f"Error deleting IAM role: {e}")


class Deploy:
    """OmniParser deployment manager."""

    @staticmethod
    def start() -> str:
        """Deploy OmniParser and return server URL.

        Returns:
            Server URL (e.g., "http://1.2.3.4:8000")
        """
        try:
            instance_id, instance_ip = configure_ec2_instance()
            if not instance_ip:
                raise RuntimeError("Failed to configure EC2 instance")

            # Trigger driver installation
            Deploy.ssh(non_interactive=True)

            # Get deployment files
            dockerfile_path = _get_dockerfile_path()
            dockerignore_path = _get_dockerignore_path()

            # Copy files to instance
            for filepath in [dockerfile_path, dockerignore_path]:
                if filepath.exists():
                    print(f"Copying {filepath.name}...")
                    subprocess.run(
                        [
                            "scp",
                            "-i", config.AWS_EC2_KEY_PATH,
                            "-o", "StrictHostKeyChecking=no",
                            str(filepath),
                            f"{config.AWS_EC2_USER}@{instance_ip}:~/{filepath.name}",
                        ],
                        check=True,
                    )

            # Connect and build
            key = paramiko.RSAKey.from_private_key_file(config.AWS_EC2_KEY_PATH)
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            try:
                ssh_client.connect(
                    hostname=instance_ip,
                    username=config.AWS_EC2_USER,
                    pkey=key,
                    timeout=30,
                )

                setup_commands = [
                    "rm -rf OmniParser",
                    f"git clone {config.REPO_URL}",
                    "cp Dockerfile .dockerignore OmniParser/",
                ]

                for cmd in setup_commands:
                    execute_command(ssh_client, cmd)

                docker_commands = [
                    f"sudo docker rm -f {config.CONTAINER_NAME} || true",
                    f"sudo docker rmi {config.PROJECT_NAME} || true",
                    f"cd OmniParser && sudo docker build --progress=plain -t {config.PROJECT_NAME} .",
                    f"sudo docker run -d -p {config.PORT}:{config.PORT} --gpus all --name {config.CONTAINER_NAME} {config.PROJECT_NAME}",
                ]

                for cmd in docker_commands:
                    execute_command(ssh_client, cmd)

                # Wait for server
                print("Waiting for server to start...")
                time.sleep(10)

                max_retries = 30
                for attempt in range(max_retries):
                    try:
                        execute_command(ssh_client, f"curl -s http://localhost:{config.PORT}/probe/")
                        break
                    except Exception:
                        if attempt < max_retries - 1:
                            print(f"Server not ready ({attempt + 1}/{max_retries}), waiting...")
                            time.sleep(10)
                        else:
                            raise RuntimeError("Server failed to start")

                server_url = f"http://{instance_ip}:{config.PORT}"
                print(f"\nDeployment complete!")
                print(f"Server URL: {server_url}")

                # Set up auto-shutdown to save costs
                create_auto_shutdown_infrastructure(instance_id)

                return server_url

            finally:
                ssh_client.close()

        except Exception as e:
            print(f"Deployment failed: {e}")
            if CLEANUP_ON_FAILURE:
                Deploy.stop()
            raise

    @staticmethod
    def status() -> None:
        """Check deployment status."""
        ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
        instances = ec2.instances.filter(
            Filters=[{"Name": "tag:Name", "Values": [config.PROJECT_NAME]}]
        )

        found = False
        for instance in instances:
            found = True
            ip = instance.public_ip_address
            if ip:
                url = f"http://{ip}:{config.PORT}"
                print(f"Instance: {instance.id} | State: {instance.state['Name']} | URL: {url}")
            else:
                print(f"Instance: {instance.id} | State: {instance.state['Name']} | No public IP")

        if not found:
            print("No instances found")

        # Check auto-shutdown status
        lambda_client = boto3.client("lambda", region_name=config.AWS_REGION)
        try:
            lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
            print(f"Auto-shutdown: Enabled ({config.INACTIVITY_TIMEOUT_MINUTES} min timeout)")
        except ClientError:
            print("Auto-shutdown: Not configured")

    @staticmethod
    def ssh(non_interactive: bool = False) -> None:
        """SSH into the running instance."""
        ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
        instances = ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )

        instance = next(iter(instances), None)
        if not instance:
            print("No running instance found")
            return

        ip = instance.public_ip_address
        if not ip:
            print("Instance has no public IP")
            return

        if not os.path.exists(config.AWS_EC2_KEY_PATH):
            print(f"Key file not found: {config.AWS_EC2_KEY_PATH}")
            return

        if non_interactive:
            subprocess.run(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-i", config.AWS_EC2_KEY_PATH,
                    f"{config.AWS_EC2_USER}@{ip}",
                    "-t", "-tt",
                    "bash --login -c 'exit'",
                ],
                check=False,
            )
        else:
            cmd = f"ssh -i {config.AWS_EC2_KEY_PATH} -o StrictHostKeyChecking=no {config.AWS_EC2_USER}@{ip}"
            print(f"Connecting: {cmd}")
            os.system(cmd)

    @staticmethod
    def stop() -> None:
        """Terminate instance and cleanup."""
        ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
        ec2_client = boto3.client("ec2", region_name=config.AWS_REGION)

        # Clean up auto-shutdown infrastructure first
        cleanup_auto_shutdown_infrastructure()

        instances = ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
            ]
        )

        for instance in instances:
            print(f"Terminating: {instance.id}")
            instance.terminate()
            instance.wait_until_terminated()
            print(f"Terminated: {instance.id}")

        try:
            ec2_client.delete_security_group(GroupName=config.AWS_EC2_SECURITY_GROUP)
            print(f"Deleted security group: {config.AWS_EC2_SECURITY_GROUP}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidGroup.NotFound":
                print(f"Error deleting security group: {e}")

    @staticmethod
    def _get_instance_ip() -> Optional[str]:
        """Get public IP of running instance."""
        ec2 = boto3.resource("ec2", region_name=config.AWS_REGION)
        instances = ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [config.PROJECT_NAME]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )
        instance = next(iter(instances), None)
        if not instance:
            print("No running instance found")
            return None
        return instance.public_ip_address

    @staticmethod
    def _run_ssh_command(command: str, timeout: int = 60) -> Optional[str]:
        """Run command on remote instance via SSH."""
        ip = Deploy._get_instance_ip()
        if not ip:
            return None
        if not os.path.exists(config.AWS_EC2_KEY_PATH):
            print(f"Key file not found: {config.AWS_EC2_KEY_PATH}")
            return None
        result = subprocess.run(
            [
                "ssh",
                "-i", config.AWS_EC2_KEY_PATH,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{config.AWS_EC2_USER}@{ip}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0 and result.stderr:
            print(f"Error: {result.stderr}")
        return result.stdout

    @staticmethod
    def logs(lines: int = 50) -> None:
        """Show container logs.

        Args:
            lines: Number of lines to show (default 50)
        """
        print(f"Fetching last {lines} lines of container logs...")
        output = Deploy._run_ssh_command(
            f"sudo docker logs {config.CONTAINER_NAME} 2>&1 | tail -{lines}"
        )
        if output:
            print(output)

    @staticmethod
    def ps() -> None:
        """Show Docker container status."""
        output = Deploy._run_ssh_command("sudo docker ps -a")
        if output:
            print(output)

    @staticmethod
    def build() -> None:
        """Build Docker image on remote instance."""
        ip = Deploy._get_instance_ip()
        if not ip:
            return

        dockerfile_path = _get_dockerfile_path()
        dockerignore_path = _get_dockerignore_path()

        # Copy Dockerfile
        for filepath in [dockerfile_path, dockerignore_path]:
            if filepath.exists():
                print(f"Copying {filepath.name}...")
                subprocess.run(
                    [
                        "scp",
                        "-i", config.AWS_EC2_KEY_PATH,
                        "-o", "StrictHostKeyChecking=no",
                        str(filepath),
                        f"{config.AWS_EC2_USER}@{ip}:~/OmniParser/",
                    ],
                    check=True,
                )

        print("Building Docker image (this may take several minutes)...")
        cmd = f"cd ~/OmniParser && sudo docker build --progress=plain -t {config.PROJECT_NAME}:latest ."
        subprocess.run(
            [
                "ssh",
                "-i", config.AWS_EC2_KEY_PATH,
                "-o", "StrictHostKeyChecking=no",
                f"{config.AWS_EC2_USER}@{ip}",
                cmd,
            ],
            check=False,
        )

    @staticmethod
    def run() -> None:
        """Start the Docker container."""
        print(f"Starting container {config.CONTAINER_NAME}...")
        Deploy._run_ssh_command(f"sudo docker rm -f {config.CONTAINER_NAME} || true")
        output = Deploy._run_ssh_command(
            f"sudo docker run -d --name {config.CONTAINER_NAME} --gpus all "
            f"-p {config.PORT}:{config.PORT} {config.PROJECT_NAME}:latest"
        )
        if output:
            print(f"Container started: {output.strip()}")
        Deploy.ps()

    @staticmethod
    def test(save_output: bool = False) -> None:
        """Test OmniParser endpoint with a synthetic image.

        Args:
            save_output: If True, save test image and results to assets/
        """
        ip = Deploy._get_instance_ip()
        if not ip:
            return

        url = f"http://{ip}:{config.PORT}"
        print(f"Testing endpoint: {url}")

        # Check health first
        try:
            import requests
            response = requests.get(f"{url}/probe/", timeout=10)
            if response.status_code != 200:
                print(f"Server not ready: {response.status_code}")
                return
            print("Server is healthy!")
        except Exception as e:
            print(f"Health check failed: {e}")
            return

        # Generate test image
        try:
            from PIL import Image, ImageDraw, ImageFont
            import base64
            import io

            img = Image.new('RGB', (400, 300), '#f0f0f0')
            draw = ImageDraw.Draw(img)

            # Draw UI elements
            draw.rectangle([30, 30, 150, 70], fill='#007bff', outline='#0056b3')
            draw.text((55, 42), 'Login', fill='white')

            draw.rectangle([30, 90, 150, 130], fill='#6c757d', outline='#545b62')
            draw.text((55, 102), 'Cancel', fill='white')

            draw.rectangle([200, 30, 370, 70], fill='#28a745', outline='#1e7e34')
            draw.text((240, 42), 'Submit', fill='white')

            draw.rectangle([200, 90, 370, 130], fill='#dc3545', outline='#bd2130')
            draw.text((245, 102), 'Delete', fill='white')

            draw.rectangle([30, 180, 370, 220], fill='white', outline='#ced4da')
            draw.text((40, 192), 'Enter username...', fill='#6c757d')

            # Encode image
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            b64 = base64.b64encode(buffer.getvalue()).decode()

            # Send to server
            print("Sending test image to server...")
            response = requests.post(
                f"{url}/parse/",
                json={'base64_image': b64},
                timeout=120
            )

            if response.status_code != 200:
                print(f"Parse failed: {response.status_code} - {response.text[:200]}")
                return

            data = response.json()
            elements = data.get('parsed_content_list', [])
            print(f"\nFound {len(elements)} elements:")
            for elem in elements:
                content = elem.get('content', '').strip()
                bbox = elem.get('bbox', [])
                elem_type = elem.get('type', 'unknown')
                print(f"  - [{elem_type}] \"{content}\" at {[f'{b:.2f}' for b in bbox]}")

            if save_output:
                # Save test image and results
                assets_dir = Path(__file__).parent.parent.parent.parent / "assets"
                assets_dir.mkdir(exist_ok=True)

                img.save(assets_dir / "test_input.png")
                print(f"\nSaved test image to assets/test_input.png")

                # Draw bounding boxes on image
                img_annotated = img.copy()
                draw = ImageDraw.Draw(img_annotated)
                w, h = img.size
                colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff']
                for i, elem in enumerate(elements):
                    bbox = elem.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
                        color = colors[i % len(colors)]
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        content = elem.get('content', '').strip()[:20]
                        draw.text((x1, y1-15), content, fill=color)

                img_annotated.save(assets_dir / "test_output.png")
                print(f"Saved annotated output to assets/test_output.png")

                # Save JSON results
                import json
                with open(assets_dir / "test_results.json", 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved results to assets/test_results.json")

        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("Install with: uv pip install pillow requests")
        except Exception as e:
            print(f"Test failed: {e}")


def main():
    """CLI entry point."""
    try:
        import fire
    except ImportError:
        raise ImportError("fire not installed. Run: uv pip install openadapt-grounding[deploy]")
    fire.Fire(Deploy)


if __name__ == "__main__":
    main()
