import os
import time
import zipfile
import requests
import json
from dotenv import load_dotenv

# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torch.nn import functional as F

from clearml import Task

# from ml_trainer import AutoTrainer
from aipmodel.model_registry import MLOpsManager

def get_dataset_download_urls(
    url: str,
    dataset_name: str,
    user_name: str,
    clearml_access_key: str,
    clearml_secret_key: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_endpoint_url: str,
    download_method: str = "presigned_urls"
):
    """
    Request presigned download URLs for a dataset.

    Returns:
        List of download URLs for .tar.gz and .csv files only
    """
    base = "http://data-ingestion-api-service.aip-mlops-service.svc.cluster.local:8169/download-dataset"

    # 1) Confirm what you’re actually hitting
    r = requests.post(f"{base}/download-dataset", json={}, timeout=10,
                  proxies={"http": None, "https": None})
    payload = {
        "dataset_name": dataset_name,
        "user_name": user_name,
        "clearml_access_key": clearml_access_key,
        "clearml_secret_key": clearml_secret_key,
        "s3_access_key": s3_access_key,
        "s3_secret_key": s3_secret_key,
        "s3_endpoint_url": s3_endpoint_url,
        "download_method": download_method
    }
    print(payload)
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=10, proxies={"http": None, "https": None})
    # response.raise_for_status()
    data = response.json()
    print(data)
    
    # Filter for .tar.gz and .csv files only
    download_urls = [
        file["download_url"]
        for file in data.get("download_info", {}).get("files", [])
        if file["filename"].endswith(".tar.gz") or file["filename"].endswith(".csv") or file["filename"].endswith('.zip')
    ]
    return download_urls[0]

# --------- ClearML task initialization --------
task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)

load_dotenv()
#  ====================== Data Registry =========================
data_model_reg_cfg= {
    #ceph related
    'CEPH_ENDPOINT': 'url',
    'CEPH_ACCESS_KEY': 'access',
    'CEPH_SECRET_KEY': 'secret',
    'CEPH_BUCKET': 'bucket',

    #clearml
    'clearml_url': 'url',
    'clearml_access_key': 'access',
    'clearml_secret_key': 'secret',
    'clearml_username': 'testdario7',
}
task.connect(data_model_reg_cfg, name='model_data_cfg')

print(data_model_reg_cfg)
os.environ['CEPH_ENDPOINT'] = data_model_reg_cfg['CEPH_ENDPOINT']
os.environ['CEPH_ACCESS_KEY'] = data_model_reg_cfg['CEPH_ACCESS_KEY']
os.environ['CEPH_SECRET_KEY'] = data_model_reg_cfg['CEPH_SECRET_KEY']
os.environ['CEPH_BUCKET'] = data_model_reg_cfg['CEPH_BUCKET']




# --------- fetch model from model registry --------
manager = MLOpsManager(
    clearml_url=data_model_reg_cfg['clearml_url'],
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
    clearml_username=data_model_reg_cfg['clearml_username']
)
# =============== training config ========
# place holders
model_id = "model_id"
save_model = False
load_model = False

cfg = {
    "task": "llm_finetuning",
    "model_name": "huggingface name",  
    "system_prompt": "system_prompts", 
    "epochs": None, 
    "batch_size": 32,
    "lr": 0.01,
    
    # defaults 
    "output_dir": "./chekpoints/medical_qa_finetune",
    "eval_steps": 50,
    "max_seq_length": 1024,

    # Model save
    "save_model": "save_model",
    "model_dir": "model/",

    "load_model": "load_model",  
    "model_dir": f"./{model_id}/",
    
    "dataset_config": {
        "name":"dataset name",
        "source": "dataset_path"
    },
    "model_config":{
        "model_reg": 'reg'
    }
}
print('before')
print(cfg)
task.connect(cfg)
print('after')
print(cfg)

model_reg = cfg["model_config"]["model_reg"]

# --------------     to load model -----------------

if load_model: 
    model_id = manager.get_model_id_by_name(model_reg)

    manager.get_model(
        model_name= model_reg,  # or any valid model ID
        local_dest=".")


url = get_dataset_download_urls(
    # url="https://api.mlops.ai-lab.ir/data/download-dataset",
    url="http://data-ingestion-api-service.aip-mlops-service.svc.cluster.local:8169/download-dataset",
    # url="https://data-ingestion-api-service:8169/download-dataset",
    dataset_name=cfg["dataset_config"]["name"],
    user_name=data_model_reg_cfg['clearml_username'],
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
    s3_access_key=data_model_reg_cfg['CEPH_ACCESS_KEY'],
    s3_secret_key=data_model_reg_cfg['CEPH_SECRET_KEY'],
    s3_endpoint_url=data_model_reg_cfg['CEPH_ENDPOINT'],
)

zip_path = "medical_qa.zip"

r = requests.get(url, stream=True)
with open(zip_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

extract_dir = "./datasets/medical_qa"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

cfg["dataset_config"]["source"] = f"{extract_dir}/medical_qa.json"
print("after data layer")




# ==== 5. Save final model ====
# save_path = "./chekpoints/medical_qa_model"
# trainer.model.save(save_path)  # Use trainer.model instead
# print(f"Model saved to {save_path}")

# trainer = AutoTrainer(config=cfg)

# trainer.run()

if save_model:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name = model_reg + "_" + str(int(time.time())),
        code_path="." , # ← Replace with the path to your model.py if you have it
    )