"""
This module connect to the Mongo DB database
"""

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import json
from pymongo import MongoClient
import os

from bson.objectid import ObjectId

import numpy as np
import pandas as pd

def connnect_and_get_collection(database_name,collection_name):
    client=MongoClient("localhost", 27017)
    db=client[database_name]
    collection=db[collection_name]
    cursor=collection.find()
    return collection, cursor

def convert_to_numpy(json_pickle_dict :dict ) -> dict:
    array=jsonpickle.decode(json.dumps(json_pickle_dict))
    
    return array.astype(int)
def clean_sacred_dict(document):
    result={}
    result["experiment"]=document["experiment"]["name"]
    result["status"]=document["status"]
    result["start_time"]=document["start_time"].strftime("%m/%d/%Y, %H:%M:%S")
    result["stop_time"]=document["stop_time"].strftime("%m/%d/%Y, %H:%M:%S")
    result["setup"]=document["setup"][5:]
    result["config"]=document["config"]
    result["cil_metrics"]=[]
    result["info"]=document["info"]
    result["host"]=document["host"]

    
    acc_metric=compute_ACC(document)
    mica_metric=compute_MICA(document)
    result["cil_metrics"].append(acc_metric)
    result["cil_metrics"].append(mica_metric)

    return result


def compute_ACC(result_dict : dict) -> dict:
    """
    Args:

    Returns:

    Raises :
    """
    flattened_logs=flatten_metrics_logs(result_dict=result_dict)
    acc={}

    for key,value in flattened_logs.items():
        acc[key]=np.mean(value)

    return {"acc":acc}

    raise NotImplemented

def flatten_metrics_logs(result_dict : dict) -> dict:
    """
    Args:

    Returns:

    Raises :
    """

    metrics=result_dict["info"]["metrics"]
    n_experiments=result_dict["config"]["data"]["n_experiments"]
    id_exps=np.arange(n_experiments).tolist()
    per_exp_metrics={}
    for id in id_exps:
        per_exp_metrics[id]=[]


    for metric_log in metrics:
        steps=metric_log["steps"]
        values=metric_log["values"]
        for i,(step,val) in enumerate(zip(steps,values)):
            per_exp_metrics[step].append(val)
    return per_exp_metrics

def compute_MICA(result_dict : dict)->dict:
    """
    Args:

    Returns:

    Raises :
    """
    flattened_logs=flatten_metrics_logs(result_dict=result_dict)
    mica={}

    for key,value in flattened_logs.items():
        mica[key]=min(value)

    return {"mica":mica}
        



def generate_results_experiments(database_name : str):
    """
    This functions connect to the database.
    Gets the metrics, configuration, and confusion matrix 

     Args : MongoDb database to query saving the experiments
     Returns : Configuration dict containing Metrics, Configuration, Configuration file, Confusion matrix
     Raises : ValueError if no data in database

    """

    dirname=os.path.dirname(os.path.realpath(__file__))
    save_path=os.path.join(dirname,database_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _,run_cursor=connnect_and_get_collection(database_name, "runs")
    metric_collection,_=connnect_and_get_collection(database_name, "metrics")
    # Get all documents
    try:
        for document in run_cursor:
            if document["status"]!="COMPLETED":
                continue
            

            # Convert to np.array the confusion matrix
            confusion_matrices=document["info"]["confusion_matrix"]["val"]
            for key,value in confusion_matrices.items():
                confusion_matrices[key]=convert_to_numpy(value).tolist()

            document["info"]["confusion_matrix"]["val"]=confusion_matrices

            #Get the metric id

            metrics_id=document["info"]["metrics"]
            for pos,value in enumerate(metrics_id):
                metric_id=value["id"]
                q=metric_collection.find_one({"_id": ObjectId(metric_id)})
                del q["_id"]
                del q["timestamps"]
                metrics_id[pos]=q
            
            document["info"]["metrics"]= metrics_id
            document["setup"]=document["experiment"]["name"].split("_")

            


            result=clean_sacred_dict(document)

            
            
            
            # name=document["experiment"]["name"]+".json"
            paths=[save_path]+document["experiment"]["name"].replace("nouvel_op","nouvelop").replace("dinov2_vits14","dinov2vits14").replace("indus_cil","induscil").split("_")
            name=os.path.join(*paths)+"/result.json"
            if not os.path.exists(os.path.join(*paths)):
                os.makedirs(os.path.join(*paths))


            with open(name, 'w') as f:
                json.dump(result,f,indent=4)
    except :
        raise ValueError



if __name__=="__main__":
    generate_results_experiments(database_name="experiments_conditions")