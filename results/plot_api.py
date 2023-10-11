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
from typing import List
from pathlib import Path

from pymfe.mfe import MFE
import torch 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from time import time

from itertools import permutations
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
    dfValidExperiments=compute_weighted_average(dfValidExperiments)


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

def crawlDataFolder(entryDirectory : str, reject : List[str]=[]):
        """
        Look in all subfolders for any files with the requested extension
        """
        foundFilesPaths=[]
        foundFilesLabels=[]
        searchedExtension =['*.pt']
        
        for pattern in searchedExtension:
            names=Path(entryDirectory).rglob(pattern)
            for path in names:
                fname = str(path)
                label=fname.replace(entryDirectory,"").split("/")[1]
                foundFilesLabels.append(label)
                foundFilesPaths.append(fname)


        return foundFilesPaths,foundFilesLabels
def computeDatasetFeatures(dataset_path : str , backbone_name : str = 'dinov2_vits14',force_regeneration=False ) -> dict :
    """
    Args: dataset_path
    Raises:
    Returns:
    """


    # Load the files in X and Y
    # Get the embeddings paths
    default_embedding_location=f"embeddings/{backbone_name}"
    embedding_path = dataset_path.replace('data', default_embedding_location)

    # Set the saving folder
    default_meta_features_location=f"metafeatures/{backbone_name}"
    default_meta_features_path=dataset_path.replace('data', default_meta_features_location)

    if not (os.path.exists(default_meta_features_path)):
        os.makedirs(default_meta_features_path)

    save_name=os.path.join(default_meta_features_path,"dataset_metafeatures_dict.json")
    

    

    # Fit the MFE

    if (os.path.exists(save_name)) and (not force_regeneration):
        with open(save_name,"r") as f:
            datasetFeatures_dict=json.load(f)
            ft=[list(datasetFeatures_dict.keys()),list(datasetFeatures_dict.values())]
            datasetFeatures=ft
            print("\n".join("{:50}  {:30}".format(x,y) for x,y in zip(ft[0],ft[1])))
    else:

        files_paths,files_labels=crawlDataFolder(embedding_path)
        named_labels=list(set(files_labels))

        id_labels=np.arange(len(named_labels),dtype=int).tolist()
                # Create an ordered dict mapping int to 
        label_dict=dict(zip(named_labels,id_labels))

        X=torch.cat(list(map(lambda x:torch.load(x).reshape(1,-1),files_paths)),axis=0)
        y=torch.tensor(list(map(lambda x:label_dict[x],files_labels)))
    

        mfe=MFE(groups="all",
                     summary=["mean"],
                     features=["ch",
                               "c2",
                               "linear_discr",
                               "naive_bayes"])
        
        # mfe=MFE(groups="all",
        #         summary=["mean"],
        #         features=["c2",
        #             "naive_bayes"])
            
        mfe.fit(X.numpy(),y.numpy())
        ft=mfe.extract()
    
        
       
        print("\n".join("{:50}  {:30}".format(x,y) for x,y in zip(ft[0],ft[1])))
        datasetFeatures=ft
        datasetFeatures_dict=dict(zip(ft[0],ft[1]))
        with open(os.path.join(save_name),"w") as f:
            json.dump(datasetFeatures_dict,f,indent=4)


    return datasetFeatures_dict

def compute_weighted_average(dataframe : pd.DataFrame) -> pd.DataFrame :
    """
    Args:
    Raises:
    Returns:
    """
    EPSILON=10**-32
    def wamica(row):
        mica=row.mica["mica"].values()
        mica=list(mica)
        res=np.mean(mica)*(1-EPSILON -np.max(mica)+np.min(mica))
        return res
    
    def waacc(row):
        acc=row.acc["acc"].values()
        acc=list(acc)
        res=np.mean(acc)*(1-EPSILON - np.max(acc)+np.min(acc))
        return res


    wamica_res=dataframe.apply(wamica,axis=1)
    waac_res=dataframe.apply(waacc,axis=1)

    new_df=pd.concat([dataframe,wamica_res,waac_res],axis=1)
    names=list(dataframe.columns)+["wamica","waacc"]
    new_df.columns=names
    
    return new_df

def numerise_datasets_with_meta_features(dataframe : pd.DataFrame, root_dataset_path : str,backbone_name : str = "dinov2_vits14") -> pd.DataFrame:
    """
    Args: 
        dataframe : the dataframe containing all experiments
        root_dataset_frame : the root dataset 
    Returns :
        result_dict : the  dataframe where the categorical feature have been replaced with the vectors corresponding to the meta features
    Raises :
    
    """

     # Load the files in X and Y
    # Get the embeddings paths
    # Set the saving folder
    datasets=dataframe.dataset

    datasets_name=datasets.unique().tolist()
    buffer={}

    for name in datasets_name:
       
        default_meta_features_location=f"metafeatures/{backbone_name}"
        default_meta_features_path=f"{root_dataset_path}/{name}/{default_meta_features_location}/dataset_metafeatures_dict.json"
        with open(default_meta_features_path,"r") as f:
            datasetFeatures_dict=json.load(f)
            buffer[name]=datasetFeatures_dict


        

    def replace(row):
        common_dict=["ch","c2","linear_discr.mean","naive_bayes.mean"]
        filter_dict={}
        for key in common_dict:
            filter_dict[key]=buffer[row["dataset"]][key]

        res=pd.Series(filter_dict)
        # res=res.loc[:,["ch","c2","linear_discr.mean","naive_bayes.mean"]]
        return res
    
    transform=dataframe.apply(replace,axis=1,result_type="expand")
    result=pd.concat([dataframe,transform],axis=1)
    result=result.drop(columns=['dataset'])
    return result

def prepare_inspection_pdp(dataframe_ : pd.DataFrame,root_path :str) -> dict :
    """
    Args: 
        dataframe : the dataframe containing all experiments
        root_path : the base path for all datasets
    Returns :
        result_dict : the cleaned dataframe as X, the target as y , the name of the categorical dataset and numerical one , models to fit
    Raises :
    """
    result={}

    dataframe=numerise_datasets_with_meta_features(dataframe=dataframe_,root_dataset_path=root_path)
    
    columns_to_drop=["experiment","acc","mica"]
    targets_columns=["wamica","waacc"]
    numerical_features_names=["ch","c2","linear_discr.mean","naive_bayes.mean"]
    X=dataframe.drop(columns=columns_to_drop+targets_columns)
    all_columns=X.columns.tolist()

    result["x"]=X

    result["y"]=[]
    for target in targets_columns:
        result["y"].append(dataframe[target])
        
    categorical_features_names=X.columns.tolist()
    categorical_features_names=X.columns.drop(numerical_features_names)
    result["cat_features_names"]=categorical_features_names

    result["models"]={"mlp":[],"gradient_boosting":[]}

    # mpl
    mlp_preprocessor = ColumnTransformer(
        transformers=[
            ("num", QuantileTransformer(n_quantiles=100), numerical_features_names),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features_names),
        ]
    )

     #gradient boosting

    hgbdt_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(), categorical_features_names),
        ("num", "passthrough", numerical_features_names),
    ],
    sparse_threshold=1,
    verbose_feature_names_out=False,
    ).set_output(transform="pandas")



    for y_ in result["y"]:
        print("Training MLPRegressor...")
        tic = time()
        mlp_model = make_pipeline(
            mlp_preprocessor,
            MLPRegressor(
                hidden_layer_sizes=(10, 10),
                learning_rate_init=0.001,
                early_stopping=True,
                random_state=0,
            ),
        )
        mlp_model.fit(X, y_)
        print(f"done in {time() - tic:.3f}s")
        print(f"Test R2 score: {mlp_model.score(X, y_):.2f}")

        result["models"]["mlp"].append(mlp_model)



        print("Training HistGradientBoostingRegressor...")
        tic = time()
        hgbdt_model = make_pipeline(
            hgbdt_preprocessor,
            HistGradientBoostingRegressor(
                categorical_features=categorical_features_names,
                random_state=0,
                max_iter=50,
            ),
        )
        hgbdt_model.fit(X, y_)
        print(f"done in {time() - tic:.3f}s")
        print(f"Test R2 score: {hgbdt_model.score(X, y_):.2f}")

        result["models"]["gradient_boosting"].append(hgbdt_model)


    common_params = {
    "subsample": 50,
    "n_jobs": 2,
    "grid_resolution": 100,
    "random_state": 0,
    }
    result["common_params"]=common_params

    permut_categorical=list(permutations(["retention","backbone"], 2))
    features_info = {
        # features of interest
        "features": permut_categorical,
        # type of partial dependence plot
        "kind": "average",
        # information regarding categorical features
        "categorical_features": categorical_features_names,
    }
    result["features_info"]=features_info
    
    return result




# def compute_h_statistic(partial_dependence_dict : dict, modality : str = "dual" ) -> dict :
#     from sklearn.datasets import load_diabetes
# from sklearn.ensemble import RandomForestRegressor
# import itertools
# import numpy as np

# diabetes = load_diabetes()
# rf = RandomForestRegressor(n_estimators=10).fit(diabetes.data, diabetes.target)

# from sklearn.inspection import partial_dependence
# univariate = {}
# for i in range(diabetes.data.shape[1]):
#     univariate[i] = partial_dependence(rf, diabetes.data, features=[i], kind='average')['average']
    
# bivariate = {}
# for i, j in itertools.combinations(range(diabetes.data.shape[1]), 2):
#     bivariate[(i, j)] = partial_dependence(rf, diabetes.data, features=[i, j], kind='average')['average']

# h = np.zeros((diabetes.data.shape[1], diabetes.data.shape[1]))
# for i, j in itertools.combinations(range(diabetes.data.shape[1]), 2):
#     h[i, j] = ((bivariate[(i, j)] - univariate[i].reshape(1, -1, 1) - univariate[j].reshape(1, 1, -1) + diabetes.target.mean() ) ** 2).sum() / ((bivariate[(i, j)] - diabetes.target.mean())** 2).sum()




#     return result
if __name__=="__main__":
   
    path="/home/mohamedphd/Documents/phd/Datasets/curated/"
    backbone_names=["dinov2_vits14","resnet18"]
    datasets=os.listdir(path)
    datasets=sorted(datasets,reverse=True)
    for backbone in backbone_names:
        for dataset in datasets:
            # dataset="kth"
            data_path=f"{path}{dataset}/data"
            print(f'\n {data_path} - {backbone}')
            dataset_features=computeDatasetFeatures(dataset_path=data_path,backbone_name=backbone)
    
    
    df=getAllValidExperiments(databaseName="representation_free")
    result=numerise_datasets_with_meta_features(df,path)
    # df=compute_weighted_average(df)
    labels=getUniqueValues(df)
    result=prepare_inspection_pdp(df,path)

    filter_dict=getFilterTemplate()
    filter_dict["dataset"]=["kth"]
    filter_dict["retention"]="None"
    filter_dict["uncertainty"]="None"
    filtered_df=filterValidExperiments(df,filter_dict)
    to_plot=formatValuesToPlottyLines(filtered_df, metric="acc")
    print(to_plot)