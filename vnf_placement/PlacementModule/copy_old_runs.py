import os 
import shutil
import yaml

def move_folder_by_id(base_dir, folder_id, destination_dir):
    #mlruns_dir = os.path.join(base_dir, "mlruns")
    #mlartifacts_dir = os.path.join(base_dir, "mlartifacts")
    #mlflow_artifacts_dir = os.path.join(base_dir, "mlflow-artifacts")
    #destination_dir = os.path.join(destination_dir)
    folders = ["mlruns", "mlartifacts", "mlflow-artifacts:"]
    #os.makedirs(destination_dir, exist_ok=True)

    for root_dir in folders:
        
        source_path = os.path.join(f"{base_dir}/{root_dir}", str(folder_id))
        destination_path = os.path.join(f"{destination_dir}/{root_dir}", str(folder_id))
        os.makedirs(destination_path, exist_ok=True)
        print(source_path)
        if os.path.exists(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok= True)
            print(f"Copied folder with ID {folder_id} to {destination_path}")

def cp_id(base_dir, folder_id, distination_id, destination_dir):

    folders = ["mlruns", "mlartifacts", "mlflow-artifacts:"]


    for root_dir in folders:
        
        source_path = os.path.join(f"{base_dir}/{root_dir}", str(folder_id))
        destination_path = os.path.join(f"{destination_dir}/{root_dir}", str(distination_id))
        os.makedirs(destination_path, exist_ok=True)
        print(source_path)
        if os.path.exists(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok= True)
            print(f"Copied folder with ID {folder_id} to {destination_path}")

def modify_meta_yaml(file_path, destination_id):
    with open(file_path, 'r') as file:
        meta_data = yaml.safe_load(file)
        
    if "experiment_id" in meta_data:
        meta_data["experiment_id"] = destination_id
        
        with open(file_path, 'w') as file:
            yaml.dump(meta_data, file, default_flow_style=False)
        
        print(f"Modified {file_path}: experiment_id updated to {destination_id}")

def copy_and_modify_batch(base_dir, source_folder_name, destination_dir, source_id,destination_id, batch_size=10):
    source_folder_path = os.path.join(base_dir, source_folder_name)
    source_folder_path = os.path.join(source_folder_path, source_id) 
    print(source_folder_path)
    destination_folder_path =  os.path.join(destination_dir, source_folder_name)
    destination_folder_path =  os.path.join(destination_folder_path, destination_id)
    if os.path.exists(source_folder_path):
        folder_list = os.listdir(source_folder_path)
        
        for i in range(0, len(folder_list), batch_size):
            batch = folder_list[i:i+batch_size]
            
            input("Appuyez sur Entrée pour copier le prochain lot...")
            
            for folder_name in batch:
                copy_and_modify(os.path.join(source_folder_path, folder_name), os.path.join(destination_folder_path, folder_name), destination_id)
    else:
        print(f"SSource folder {source_folder_path} not found")

def copy_and_modify(source_path, destination_path, destination_id):
    if os.path.exists(source_path):
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            
            meta_yaml_path = os.path.join(destination_path, "meta.yaml")
            if os.path.exists(meta_yaml_path):
                modify_meta_yaml(meta_yaml_path, destination_id)
                print("done")
            else:
                print(f"meta.yaml not found in {destination_path}")
        else:
            print("Not a directory")
    else:
        print(f"Source folder {source_path} not found")        

base_dir = "/home/ryad/rl-multi-domain-for-multi-placement/vnf_placement/PlacementModule/volume_mlflow/mlflow/"
destination_dir = "/home/ryad/rl-multi-domain-for-multi-placement/vnf_placement/PlacementModule/volume_mlflow/mlflow/"

id = "765116458666931120"
destination_id = "433567329478374211"
copy_and_modify_batch(base_dir, "mlruns", destination_dir, id, destination_id, batch_size=10)
