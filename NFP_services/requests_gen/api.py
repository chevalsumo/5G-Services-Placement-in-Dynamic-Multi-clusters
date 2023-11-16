import re 
from fastapi import FastAPI
import os

charts_path = "/home/code/charts"#os.environ.get("CHARTS_PATH")

def generate_slice_req(nsid, type, CHARTS_PATH):
    """
    Generate a slice request (SR) of a specific type 

    Args:
        - nsid (str): The ID of slice request associated to SR
        - type (str):  Type of the SR (CN/RAN/RAN+CN)
        - CHARTS_PATH (str): Path to 5G NF Helm charts
    """
    #CHARTS_PATH = "../../towards5gs-helm/charts"    
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
        "DN": {
            "default_values": f"{CHARTS_PATH}/nginx",
            "resources": {
                "requests": {
                    "cpu": "10m",
                    "memory": "40Mi"
                },
                "limits": {
                    "cpu": "5m",
                    "memory": "10Mi"
                }
            }
        },
    }
    if type == "CN-RAN":
        pass

        #return SR
    elif type == "CN":
        del SR["gnb"]
        del SR["ue"]
        pass
        #return SR
    elif type == "RAN":
        SR = {
            "gnb": SR["gnb"],
            "ue": SR["ue"]
        }
    
    document = {
        "NSID" : nsid,
        "nfs" : {}
    }
    for nf, data_nf in SR.items():
        rs = data_nf["resources"]
        if "limits" not in rs :
            document["nfs"][nf] = {
                        'cpu_limits': 0,
                        'cpu_requests': 0,
                        'ram_limits': 0,
                        'ram_requests': 0
                                    } 
            for _, _data_nf in rs.items():
                document["nfs"][nf]['cpu_limits'] +=  int(re.search(r"\d+", _data_nf['limits']['cpu']).group())
                document["nfs"][nf]['cpu_requests']+= int(re.search(r"\d+", _data_nf['requests']['cpu']).group())
                document["nfs"][nf]['ram_limits']+= int(re.search(r"\d+", _data_nf['limits']['memory']).group())
                document["nfs"][nf]['ram_requests']+= int(re.search(r"\d+", _data_nf['requests']['memory']).group())
                                        
        else :
            document["nfs"][nf] = {
                        'cpu_limits': int(re.search(r"\d+", rs['limits']['cpu']).group()),
                        'cpu_requests': int(re.search(r"\d+", rs['requests']['cpu']).group()),
                        'ram_limits': int(re.search(r"\d+", rs['limits']['memory']).group()),
                        'ram_requests': int(re.search(r"\d+", rs['requests']['memory']).group())
                                  }
    if type =="v2x":
        max_delay = 15 
        delay = {
            "gnb": {"source": "ue", "delay": 4.55},   # 1
            "upf": {"source": "gnb", "delay": 2.928}, # 2
            "DN": {"source": "upf", "delay": 2.29},
            "smf": {"source": "upf", "delay": max_delay},
            "udm": {"source": "nrf", "delay": max_delay},
            "udr": {"source": "nrf", "delay": max_delay},
            "pcf": {"source": "nrf", "delay": max_delay},
            "nssf": {"source": "nrf", "delay": max_delay},
            "ausf": {"source": "nrf", "delay": max_delay},
            "amf": {"source": "nrf", "delay": max_delay},  # 3
            "nrf" : {"source": "ue", "delay": max_delay},  # 4
            "webui": {"source": "nrf", "delay": max_delay}
        }
        return SR, document, delay
    return SR, document, None

app = FastAPI()
api_path = os.environ.get('API_PATH')
@app.get("/rg/{_id}/{_type}")
def request_generator(_id, _type):
    sr, doc, delay = generate_slice_req(_id, _type, charts_path)
    print(sr)
    return {
        "sr" : sr,
        "doc": doc,
        "delay": delay,
    }
@app.get("/rg/doc/{_id}/{_type}")
def doc_request_generator(_id, _type):
    sr, doc, delay = generate_slice_req(_id, _type, charts_path)
    print(sr)
    return {
        "doc": doc,
        "delay": delay,
    }