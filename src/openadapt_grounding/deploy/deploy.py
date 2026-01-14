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
"""

import os
import subprocess
import time
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

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic
    from pydantic import BaseSettings


CLEANUP_ON_FAILURE = False


class Config(BaseSettings):
    """Configuration settings for deployment."""

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"

    PROJECT_NAME: str = "omniparser"
    REPO_URL: str = "https://github.com/microsoft/OmniParser.git"
    # Deep Learning AMI GPU PyTorch 2.3.1 (Ubuntu 22.04)
    AWS_EC2_AMI: str = "ami-06835d15c4de57810"
    AWS_EC2_DISK_SIZE: int = 128  # GB
    AWS_EC2_INSTANCE_TYPE: str = "g4dn.xlarge"  # T4 16GB $0.526/hr
    AWS_EC2_USER: str = "ubuntu"
    PORT: int = 8000  # FastAPI port
    COMMAND_TIMEOUT: int = 600  # 10 minutes

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def CONTAINER_NAME(self) -> str:
        return f"{self.PROJECT_NAME}-container"

    @property
    def AWS_EC2_KEY_NAME(self) -> str:
        return f"{self.PROJECT_NAME}-key"

    @property
    def AWS_EC2_KEY_PATH(self) -> str:
        return f"./{self.AWS_EC2_KEY_NAME}.pem"

    @property
    def AWS_EC2_SECURITY_GROUP(self) -> str:
        return f"{self.PROJECT_NAME}-SecurityGroup"


config = Config()


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


def main():
    """CLI entry point."""
    try:
        import fire
    except ImportError:
        raise ImportError("fire not installed. Run: uv pip install openadapt-grounding[deploy]")
    fire.Fire(Deploy)


if __name__ == "__main__":
    main()
