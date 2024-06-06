"""
This module help build experiments config files and serve them to training based on a training plan

"""

import json
import os
from plugins import plugins_dict
import copy

def generate_save_conditions_experiments(experiment_name,
                                         representation=['Repr_Free'],
                                         scenarii = ['cil','indus_cil'],):
    dirname=os.path.dirname(os.path.realpath(__file__))
    save_path=os.path.join(dirname,experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    default_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),"default_skeleton.json")

    # knowledge_incorporation=['CE','L2','Dirichlet CE']
    # knowledge_retention=["None","KD",'Dynamic Arch',"big buffer"]
    # Bias_Mitigation=["None","wa","bic","finetuning"]
    # uncertainty_reduction=["None","dirichlet","conformal"]
    base_path="/home/mohamedphd/Documents/phd/"


    paths=[base_path+"Datasets/curated/kth",
            base_path+"Datasets/curated/magnetic",
            base_path+"Datasets/curated/mvtec",
            base_path+"Datasets/curated/dagm",
            base_path+"Datasets/curated/easy",
            base_path+"Datasets/curated/nouvel_op"]


    
    with open(default_path, 'r') as f:
        default_dict=json.load(f)




    # Update working json for representation
    # Impact the model and data
    free_representation_choices=["resnet18","dinov2_vits14"]
    free_representation_backbones=["None"]

    fixed_representation_choices=["mlp"]
    fixed_representation_backbones=["resnet18","dinov2_vits14"]



    conversion_dict={'Repr_Free':[(free_representation_choices,free_representation_backbones)]
                    ,'Repr_Fixed':[(fixed_representation_choices,fixed_representation_backbones)],
                    'Incorporation': [plugins_dict['CrossEntropyOperation']],
                    'Retention':[plugins_dict['KnowledgeDistillationOperation']],
                    'Bias':[plugins_dict['FinetuneOperation']],
                    'Uncertainty' : [None ]}


    configs=[]
    working_json=copy.deepcopy(default_dict)
    for scenario in scenarii:
        for path in paths:
            for given_representation in representation:
                for model in conversion_dict[given_representation][0][0]:
                    for backbone in conversion_dict[given_representation][0][1]:
                        for knowledge_incor in conversion_dict["Incorporation"]:
                            for ret in conversion_dict["Retention"]:
                                for bias in conversion_dict["Bias"]:
                                    for uncert in conversion_dict["Uncertainty"]:
                                        
                                        working_json=copy.deepcopy(default_dict)
                                        working_json["data"]["scenario"]=scenario
                                        working_json["data"]["data_path"]=path
                                        working_json["data"]["dataset_name"]=path.split("/")[-1]
                                        working_json["model"]["model_type"]=model
                                        working_json["data"]["backbone"]=backbone
                                        kn_name='None'
                                        ret_name='None'
                                        bias_name='None'
                                        unc_name='None'


                                        if knowledge_incor is not None: 
                                            working_json["plugins"].append(knowledge_incor().config_template)
                                            kn_name=working_json["plugins"][-1]['name']
                                        if ret is not None:
                                            working_json["plugins"].append(ret().config_template)
                                            ret_name=working_json["plugins"][-1]['name']
                                        if bias is not None: 
                                            working_json["plugins"].append(bias().config_template)
                                            bias_name=working_json["plugins"][-1]['name']
                                        if uncert is not None:
                                            working_json["plugins"].append(uncert().config_template)
                                            unc_name=working_json["plugins"][-1]['name']
                                        
                                        configs.append(working_json)
                                        # name="_".join([model,backbone,path.split("/")[-1],scenario,kn_name,ret_name,bias_name,unc_name])
                                        name="_".join([path.split("/")[-1].replace("_",""),
                                                       scenario.replace("_",""),
                                                       model.replace("_",""),
                                                       backbone.replace("_",""),
                                                       kn_name.replace('Operation',""),
                                                       ret_name.replace('Operation',""),
                                                       bias_name.replace('Operation',""),
                                                       unc_name.replace('Operation',"")])
                                        name=name+".json"


                                        with open(os.path.join(save_path,name), 'w') as f:
                                            json.dump(working_json,f,indent=4)


    return save_path

if __name__=="__main__":
    generate_save_conditions_experiments(experiment_name="Conditions_Of_IL_Experiments_Repr_Fixed",representation=['Repr_Fixed'])                                        
                                        


