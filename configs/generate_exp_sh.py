"""
This module buil sh files to run experiments in a queue

"""

import os
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
def checkExperimentsExistence(config_name, collection) -> bool:
    """
    Args

    Returns

    Raises
    """
    q=collection.find_one({"experiment.name": config_name})
    if q is not None :
        return True
    else: return False

def generate_sh_script(path,savingDatabase=None):
    files=["#!/usr/bin/env bash\n\n","cd \"/home/mohamedphd/Documents/phd/clmixer\"\n"]
    savepath=path+'.sh'
    config_files=os.listdir(path)
    for config_file in config_files:
        abspath=os.path.abspath(os.path.join(path,config_file))

        if savingDatabase is not None:
            #Load the database
            collection,cursor= connnect_and_get_collection(savingDatabase,collection_name="runs")

            if checkExperimentsExistence(config_file[:-5],collection):
                continue
            
        command=f"python3 training.py with \"{abspath}\" -D -p -n \"{config_file[:-5]}\" --force\n"
        files.append(command)
    
    with open(savepath,'w') as f:
        f.writelines(files)

if __name__=="__main__":
    generate_sh_script("/home/mohamedphd/Documents/phd/clmixer/configs/Conditions_Of_IL_Experiments_Repr_Fixed")
