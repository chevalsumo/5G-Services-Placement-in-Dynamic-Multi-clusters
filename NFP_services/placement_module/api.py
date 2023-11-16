from fastapi import FastAPI
from BD_management import *
from pth_metrics import *
app = FastAPI()
CONNECTION_STRING = "mongodb://mongodb:27017"
thanos_url = "http://c0-worker:31897"
 
@app.get("/db/infra_limits")
def read_root():
    return {"infra_db": get_infra_from_BD(CONNECTION_STRING)}

@app.get("/pth/clusters_all_loaded")
def clusters_all_loaded():
    namespaces = get_all_slices(CONNECTION_STRING)
    clusters = get_list_clusters()
    print(namespaces, flush= True)
    print(clusters, flush= True)
    load = get_clusters_loaded_consumption(thanos_url, clusters, namespaces, cpu_rate="1m")
    print(load, flush= True)
    return {"loaded_clusters": load}


