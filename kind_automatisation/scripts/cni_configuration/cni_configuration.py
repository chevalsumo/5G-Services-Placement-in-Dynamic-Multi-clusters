import subprocess
import yaml

def CNI_tigera_install(cls_name, pod_subnet):
    """
    Install the Tigera CNI plugin on a Kubernetes cluster.

    This function installs the Tigera CNI plugin on the specified Kubernetes cluster identified by the specified cluster name.
    The `pod_subnet` parameter specifies the value of the pod_subnet for the cluster.

    Args:
    - cls_name (str): The name of the Kubernetes cluster to install Tigera CNI on.
    - pod_subnet (str): The value of pod_subnet for the cluster.

    Returns:
    - None
    """
    # The YAML file to apply for the CNI of "cls_name" cluster
    tigera_config = {
        'apiVersion': 'operator.tigera.io/v1',
        'kind': 'Installation',
        'metadata': {
            'name': 'default'
        },
        'spec': {
            'calicoNetwork': {
                'ipPools': [
                    {
                        'blockSize': 26,
                        'cidr': pod_subnet,
                        'encapsulation': 'VXLANCrossSubnet',
                        'natOutgoing': 'Enabled',
                        'nodeSelector': 'all()'
                    }
                ],
                'containerIPForwarding': 'Enabled'
            }
        }
    }
    config_path = f'../../config/cni_configuration/cni_config_{cls_name}.yaml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(tigera_config, f)

    subprocess.run(['kubectl', 'create', '-f', 'https://raw.githubusercontent.com/projectcalico/calico/v3.25.0/manifests/tigera-operator.yaml', "--context", f'kind-{cls_name}'])
    subprocess.run(['kubectl', 'apply', '-f', config_path, "--context", f'kind-{cls_name}'])
    #subprocess.run(['kubectl', 'wait', '--for=condition=Ready', '--timeout=600s', 'pod', '-A', '--all', "--context", f'kind-{cls_name}'])

def CNI_multus_install(cls_name):
    """
    Deploy Multus CNI plugin to a Kubernetes cluster.

    This function deploys the Multus CNI plugin to a Kubernetes cluster identified by the specified cluster name.
    Multus CNI is a container network interface plugin that enables attaching multiple network interfaces to pods.

    Args:
    - cls_name (str): The name of the Kubernetes cluster to deploy Multus CNI to.

    Returns:
    - None
    """
    config_path = f'../../multus-cni/deployments/multus-daemonset-thick.yml'
    subprocess.run(['kubectl', 'apply', '-f', config_path, "--context", f'kind-{cls_name}'])