import sys
import os
import yaml
from BD_management import *


def adapte_mutus_subnets(CHART, values, subnet, n6_subnet = None):
    """
    Modifies the subnets of N2/N3/N6 networks used in network function charts (AMF/SMF/GNB/UPF) deployments for each slice.

    Args:
    - CHART (str): The type of network function chart ("amf", "smf", "upf", "gnb").
    - values (dict): The dictionary containing the network function values.
    - subnet (int): The subnet value to use for (N2/N3/N4) networks.
    - n6_subnet (int, optional): The subnet value to use for N6 network. Default is None.

    Returns:
    - values (dict): The modified dictionary containing the updated network function values.
    """

    if CHART == "amf":
        # Modify the values for AMF chart
        if "global" not in values:
            values["global"] = {}

        if "amf" not in values["global"]:
            values["global"]["amf"] = {}

        if "n2if" not in values["global"]["amf"]:
            values["global"]["amf"]["n2if"] = {}

        if "n2network" not in values["global"]:
            values["global"]["n2network"] = {}

        values["global"]["amf"]["n2if"]["ipAddress"] = f"10.{subnet}.249"
        values["global"]["n2network"]["subnetIP"] = f"10.{subnet}.248"
        values["global"]["n2network"]["gatewayIP"] = f"10.{subnet}.254"
        values["global"]["n2network"]["excludeIP"] = f"10.{subnet}.254"
    elif CHART == "smf":
        # Modify the values for SMF chart
        if "global" not in values:
            values["global"] = {}
        if "smf" not in values["global"]:
            values["global"]["smf"] = {}
        if "n4if" not in values["global"]["smf"]:
            values["global"]["smf"]["n4if"] = {}
        if "n4network" not in values["global"]:
            values["global"]["n4network"] = {}

        values["global"]["smf"]["n4if"]["ipAddress"] = f"10.{subnet}.244"
        values["global"]["n4network"]["subnetIP"] = f"10.{subnet}.240"
        values["global"]["n4network"]["gatewayIP"] = f"10.{subnet}.246" 
        values["global"]["n4network"]["excludeIP"] = f"10.{subnet}.246" 
    elif CHART == "upf":
        # Modify the values for UPF chart
        if "global" not in values:
            values["global"] = {}
        if "upf" not in values:
            values["upf"] = {}
        if "n4if" not in values["upf"]:
            values["upf"]["n4if"] = {}
        if "n3if" not in values["upf"]:
            values["upf"]["n3if"] = {}
        if "n6if" not in values["upf"]:
            values["upf"]["n6if"] = {}
        if "n3network" not in values["global"]:
            values["global"]["n3network"] = {}
        if "n4network" not in values["global"]:
            values["global"]["n4network"] = {}
        if "n6network" not in values["global"]:
            values["global"]["n6network"] = {}

        values["upf"]["n3if"]["ipAddress"] = f"10.{subnet}.233"
        values["upf"]["n4if"]["ipAddress"] = f"10.{subnet}.241"
        values["upf"]["n6if"]["ipAddress"] = f"10.{n6_subnet}.12"

        values["global"]["n3network"]["subnetIP"] = f"10.{subnet}.232"
        values["global"]["n3network"]["gatewayIP"] = f"10.{subnet}.238"
        values["global"]["n3network"]["excludeIP"] = f"10.{subnet}.238"

        values["global"]["n4network"]["subnetIP"] = f"10.{subnet}.240"
        values["global"]["n4network"]["gatewayIP"] = f"10.{subnet}.246" 
        values["global"]["n4network"]["excludeIP"] = f"10.{subnet}.246"

        values["global"]["n6network"]["subnetIP"] = f"10.{n6_subnet}.0"
        values["global"]["n6network"]["gatewayIP"] = f"10.{n6_subnet}.1" 
        values["global"]["n6network"]["excludeIP"] = f"10.{n6_subnet}.254"
    elif CHART == "gnb":
        # Modify the values for GNB UERANSIM chart
        if "global" not in values:
            values["global"] = {}
        if "gnb" not in values:
            values["gnb"] = {}

        values["global"]["n2network"] = {"subnetIP" : f"10.{subnet}.248",
                                         "gatewayIP" : f"10.{subnet}.254",
                                         "excludeIP" : f"10.{subnet}.254"}

        values["global"]["n3network"] = {"subnetIP" : f"10.{subnet}.232",
                                         "gatewayIP" : f"10.{subnet}.238",
                                         "excludeIP" : f"10.{subnet}.238"}

        values["gnb"]["n2if"] = {"ipAddress" : f"10.{subnet}.250"}
        values["gnb"]["n3if"] = {"ipAddress" : f"10.{subnet}.236"}
        values["gnb"]["amf"] = {"n2if":{"ipAddress" : f"10.{subnet}.249"}}
    return values
def install_NF(CHARTS_PATH, NAMESPACE, context, CHART, ressources, subnet, n6_subnet = None):
    """
    Install NF chart in a the namespace of a specefic cluster.

    Args:
    - CHARTS_PATH (str): The path to the charts directory.
    - NAMESPACE (str): The namespace to deploy the chart.
    - subnet (int): The subnet value to use for (N2/N3/N4) networks.
    - n6_subnet (int, optional): The subnet value to use for N6 network. Default is None.
    - context (str): The context to use for the Kubernetes cluster.
    
    - CHART (str): The name of the chart to install.
    """
    # Create custom values to use in the NFs charts to use the exported services (nrf/mongoDB) by submariner
    values = {
        "global": {
            "nrf": {
                "service": {
                    "name": f"nrf-nnrf.{NAMESPACE}.svc.clusterset.local"
                }
            }
        },
        "mongodb": {
            "service": {
                "name": f"mongodb.{NAMESPACE}.svc.clusterset.local"
            }
        },
        
    }
    
    


    if CHART != "nrf":
        values = {
            "global": {
                "nrf": {
                    "service": {
                        "name": f"nrf-nnrf.{NAMESPACE}.svc.clusterset.local"
                    }
                }
            },
            "mongodb": {
                "service": {
                    "name": f"mongodb.{NAMESPACE}.svc.clusterset.local"
                }
            },
           CHART : {
               "resources" : ressources
           }
        }       
    else : 
        values = {
            "nrf" : {
                "resources" : ressources["nrf"]
            },
            "mongodb" : {
                "resources" : ressources["mongodb"]  
            }
       }

    values = adapte_mutus_subnets(CHART, values, subnet, n6_subnet = n6_subnet)
    VALUES_PATH = "../../config/nfs_placement/nfs_values.yaml"
    with open(VALUES_PATH, "w") as f:
        yaml.dump(values, f)
    values_file = f"-f {VALUES_PATH}"
    
    helm_command = f"helm install {CHART} {CHARTS_PATH} -n {NAMESPACE} --kube-context {context} {values_file} --create-namespace"
    os.system(helm_command)

def uninstall_NF(NAMESPACE, context, CHART):
    """
    Uninstalls a chart.

    Args:
    - NAMESPACE (str): The namespace where the chart is deployed.
    - context (str): The context to use for the Kubernetes cluster.
    - CHART (str): The name of the chart to uninstall.
    """
    helm_command = f"helm -n {NAMESPACE} --kube-context {context} uninstall {CHART}"
    os.system(helm_command)


if __name__ == "__main__":



    CHARTS_PATH = "../../towards5gs-helm/charts"  
    # Chart of a Slice Request that we will deploy :
    SR = {
        "nrf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-nrf",
            "resources": {
                "nrf": {
                    "requests": {
                        "cpu": "40m",
                        "memory": "64Mi"
                    },
                    "limits": {
                        "cpu": "50m",
                        "memory": "128Mi"
                    }
                },
                "mongodb": {
                    "requests": {
                        "cpu": " 70m",
                        "memory": "200Mi"
                    },
                    "limits": {
                        "cpu": "100m",
                        "memory": "256Mi"
                    }
                }
            }
        },
        "udr": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-udr",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "udm": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-udm",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "ausf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-ausf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "nssf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-nssf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "amf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-amf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "pcf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-pcf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "smf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-smf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "webui": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-webui",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "upf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-upf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "ue": {
            "default_values": f"{CHARTS_PATH}/ueransim/charts/ue",
            "resources": {
                "requests": {
                    "cpu": "50m",
                    "memory": "128Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "200Mi"
                }
            }
        },
        "gnb": {
            "default_values": f"{CHARTS_PATH}/ueransim/charts/gnb",
            "resources": {
                "requests": {
                    "cpu": "50m",
                    "memory": "256Mi"
                },
                "limits": {
                    "cpu": "60m",
                    "memory": "300Mi"
                }
            }
        },
    }

    NFs_PLACEMENT = {
        "nrf"  : "kind-c1",
        "udr"  : "kind-c1",
        "udm"  : "kind-c1",
        "ausf" : "kind-c1",
        "nssf" : "kind-c1",
        "amf"  : "kind-c1",
        "pcf"  : "kind-c2",
        "smf"  : "kind-c2",
        "webui": "kind-c2",
        "upf"  : "kind-c2",
        "ue"   : "kind-c2",
        "gnb"  : "kind-c2"
        }
    #NFs_PLACEMENT = { "gnb"  : "kind-c1",} 
    # "gnb"  : "kind-c1"}
    NAMESPACE = sys.argv[1]
    OPERATION = sys.argv[2]
    i = 3
    subnet, n6_subnet = f"100.{50+i}",f"100.{100+i}"
    for chart, context in NFs_PLACEMENT.items():
        if OPERATION == "uninstall":
            helm_command = f"helm -n {NAMESPACE} --kube-context {context} uninstall {chart}"
            os.system(helm_command)
        elif OPERATION == "install":
            install_NF(SR[chart]["default_values"], NAMESPACE, context, chart, SR[chart]["resources"], subnet, n6_subnet= n6_subnet )


    # Exporter les services MongoDB et NrF sur submariner 
    CONNECTION_STRING = "mongodb://localhost:27017"

    if OPERATION == "install" :
        mongo_db_export_command = f"subctl export service -n {NAMESPACE} mongodb --context {NFs_PLACEMENT['nrf']}"
        nrf_export_command = f"subctl export service -n {NAMESPACE} nrf-nnrf --context {NFs_PLACEMENT['nrf']}"
        os.system(mongo_db_export_command)
        os.system(nrf_export_command)
        add_slice_req(CONNECTION_STRING, NAMESPACE, "CN/RN")
        add_nfs_slice(CONNECTION_STRING, NAMESPACE, SR)
    elif OPERATION == "uninstall":
        remove_slice_request(CONNECTION_STRING, NAMESPACE)
        remove_nfs_slice(CONNECTION_STRING, NAMESPACE)