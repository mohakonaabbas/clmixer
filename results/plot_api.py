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
    acc_metric=compute_ACC(document)
    mica_metric=compute_MICA(document)

    result["experiment"]=document["experiment"]["name"]
    result["dataset"]=document["config"]["data"]["dataset_name"]
    result["backbone"]=document["config"]["data"]["backbone"]
    result["architecture"]=document["config"]["model"]["model_type"]

    [incorporation,retention,bias,uncertainty]=document["setup"][-4:]
    result["incorporation"]=incorporation
    result["retention"]=retention
    result["bias"]=bias
    result["uncertainty"]=uncertainty

    result["acc"]=acc_metric
    result["mica"]=mica_metric
    result["scenario"]=document["config"]["data"]["scenario"]
    
    return result

def getFilterTemplate():
    template={}
    for key in ["experiment",
                "dataset",
                "backbone",
                "architecture",
                "incorporation",
                "retention",
                "bias",
                "uncertainty",
                "scenario"]:
        template[key]=[]
    return template


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
    
    # Create the df placeholder
    df=pd.DataFrame()
    
    # Get all documents
    # try:
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
        corrected_name=document["experiment"]["name"].replace("nouvel_op","nouvelop").replace("dinov2_vits14","dinov2vits14").replace("indus_cil","induscil").split("_")
        document["setup"]=corrected_name


        result=clean_sacred_dict(document)
        # Append Rows to Empty DataFrame.
        df = df.append(result, ignore_index = True)
    return df


    # except :
    #     raise ValueError

#APIS

def getAllValidExperiments(databaseName : str) -> pd.DataFrame:
    """
    Args:
        databaseName : The name of the MongoDB database where all the experiments are stored

    Returns:
        dfValidExperiments : a pandas Dataframe that return all the experiments stored in the database

    Raises:
    """
    dfValidExperiments=generate_results_experiments(databaseName)


    return dfValidExperiments

def filterValidExperiments(dfValidExperiments : pd.DataFrame, filterDict : dict = {}) -> pd.DataFrame :
    """
    Args:
        dfValidExperiments : a pandas Dataframe that stores all valid experiments stored in the database
        filterDict : a dict with keys as the filtering key (Database, Backbone, ...) and the values queried (all, None, ...)

    Returns:
    filteredValidExperiments : a filtered pandas dataframe

    Raises:
        ValueError if no data is left after filtering
    """

    filteredValidExperiments=dfValidExperiments.copy()

    for key, value in filterDict.items():
        if value == []: continue

        if not isinstance(value,list):
            value=[value]
        positive_mask=pd.Series(np.zeros(filteredValidExperiments.shape[0])!=0)
        for val in value:
            #Concat the result here
            current_filter=filteredValidExperiments[key]==val
            current_filter=current_filter.reset_index(drop=True)
            # print(current_filter.shape)
            positive_mask=positive_mask.add(current_filter)

        filteredValidExperiments=filteredValidExperiments.loc[positive_mask]
        filteredValidExperiments=filteredValidExperiments.reset_index(drop=True)

    if filteredValidExperiments.shape[0]>0:
        return filteredValidExperiments
    else:
        raise Exception("No data left after filtering")

def formatValuesToPlottyLines(experiments : pd.DataFrame,
                              metric : str = "acc") -> dict:
    """
    Args:
        experiments : a filtered pandas dataframe
        
    Returns:
    experiment : a dict representing a ploty figure

    Raises:
        ValueError if no data is left after filtering

    """
    datum={"x":None,
           "y": None,
           "mode":'line+markers',
            "name":None}
    
    df=experiments.copy()
    
    def formatter(row):
        metric_value=row[metric]
        name=row["experiment"]
        values=list(metric_value[metric].values())
        x=np.arange(len(values)).tolist()
        result={"x":x,
           "y": values,
           "mode":'lines+markers',
            "name":name}
        return result

    plots=df.apply(formatter,axis=1)



    
     
    return plots


def getUniqueValues(dfValidExperiments : pd.DataFrame) -> dict:
    """
    Args:

    Returns:

    raises:
    
    """

    # Get the colums names
    columns=dfValidExperiments.columns.tolist()
    columns.remove("acc")
    columns.remove("mica")
    items= list(map(lambda x : dfValidExperiments[x].unique().tolist(),columns))

    items_list=dict(zip(columns,items))
    return items_list

if __name__=="__main__":
    df=getAllValidExperiments(databaseName="experiments_conditions")
    labels=getUniqueValues(df)
    filter_dict=getFilterTemplate()
    filter_dict["dataset"]=["kth"]
    filter_dict["retention"]="None"
    filter_dict["uncertainty"]="None"
    filtered_df=filterValidExperiments(df,filter_dict)
    to_plot=formatValuesToPlottyLines(filtered_df, metric="acc")
    print(to_plot)