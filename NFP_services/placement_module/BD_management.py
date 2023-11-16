from pymongo import MongoClient
import re 


def get_offset_subnets(mongoDBURL):
    """
    
    """
    client = MongoClient(mongoDBURL)
    offsets = client["NSlies"]["offsets"]
    offset_subnets_2 = offsets.find_one({'type': 'offset_subnets_2'})
    offset_subnets_1 = offsets.find_one({'type': 'offset_subnets_1'})
        
    return offset_subnets_1['value'], offset_subnets_2['value']

def set_offset_subnets(mongoDBURL, i_1, i_2):
    """
    
    """
    client = MongoClient(mongoDBURL)
    offsets = client["NSlies"]["offsets"]
    critere = {"type":"offset_subnets_1"}
    offsets.update_one(
            {"type": "offset_subnets_1"},
            {"$set": {"value" : i_1}},
            upsert=True
        )
    offsets.update_one(
            {"type": "offset_subnets_2"},
            {"$set": {"value" : i_2}},
            upsert=True
        )
    return None 

def get_all_slices(mongoDBURL):
    """
    Retrieve all slices IDs from MongoDB

    Args:
        - mongoDBURL (str): The URL of the MongoDB

    Returns:
        - list: A list of all nsid present in the collection
    """
    client = MongoClient(mongoDBURL)
    slices_requests = client["NSlies"]['slices_requests']
    all_slices = slices_requests.distinct("ID")
    return all_slices

def remove_slice_request(mongoDBURL, nsid):
    """
    Remove a slice request from the MongoDB NSlices collection
    in the slices_requests collection

    Args:
        - mongoDBURL (str): The URL of the MongoDB
        - nsid (str): The ID of the slice request (nsid) to remove
    """
    client = MongoClient(mongoDBURL)
    slices_requests = client["NSlies"]['slices_requests']
    slices_requests.delete_one({"ID": nsid})

def add_slice_req(mongoDBURL, id, type):
    """
    Add a slice request (nsid) to the monogoDB NSlices
    In the slices_requests collection

    Args:
        - mongoDBURL (str): The URL of the MongoDB
        - nsid (str): The ID of slice request associated to SR
        - type (str):  Type of the SR (CN/RAN/RAN+SR)
    """
    client = MongoClient(mongoDBURL)
    document = {
        "ID" : id,
        "type": type
    }
    slices_requests = client["NSlies"]['slices_requests']
    slices_requests.insert_one(document)

def add_nfs_slice(mongoDBURL, document, getDoc = False):
    """
    Add the NFs of a slice request (nsid) to the monogoDB NSlices
    In the network_functions collection

    Args:
        - mongoDBURL (str): The URL of the MongoDB
        - nsid (str): The ID of slice request associated to SR
        - sr (dict):  a dictionary containing (nsid) network functions and their configuration 
        - getDoc (boolean): a Boolean to determine whether to return the document or save it in the DB (for the training phase) 
     """
    client = MongoClient(mongoDBURL)
    network_functions = client["NSlies"]['network_functions']

            
    if getDoc :
        return document
    else:    
        network_functions.insert_one(document)
    


def remove_nfs_slice(mongoDBURL, nsid):
    """
    Remove the NFs of a slice request from the MongoDB NSlices collection
    in the network_functions collection

    Args:
        - mongoDBURL (str): The URL of the MongoDB
        - nsid (str): The ID of the slice request (nsid) to remove the NFs from
    """
    client = MongoClient(mongoDBURL)
    network_functions = client["NSlies"]['network_functions']
    network_functions.delete_one({"NSID": nsid})

def get_infra_from_BD(mongoDBURL):
    client = MongoClient(mongoDBURL)
    infra = client["NSlies"]['infra']
    clusters = infra.find()
    json_docs = [{k: v for k, v in doc.items() if k != '_id'}for doc in clusters]
    return json_docs


def get_available_resources(limits, loaded):
    available = {}
    print(limits)
    for cls in limits:
        if "ID" in cls: 
            cluster_id = cls["ID"]
            if cluster_id in loaded :
                available[cluster_id] = {}
                available[cluster_id]["RAM"] = cls["ram"] - loaded[cluster_id]["RAM"]
                available[cluster_id]["CPU"] = cls["cpu"] - loaded[cluster_id]["CPU"]
    return available

if __name__ == "__main__": 
    

  
    CONNECTION_STRING = "mongodb://localhost:27017"
    client = MongoClient(CONNECTION_STRING)

    id = "orange"
    type = "CN"

    #add_slice_req(client, id, type)
    nsid = "nokia"
    #sr = generate_slice_req(nsid, "CN/RAN")
    #print(sr)
    #add_nfs_slice(client, nsid, sr)
    #add_slice_req(CONNECTION_STRING, "test","CN")
    #get_infra_from_BD(CONNECTION_STRING)
