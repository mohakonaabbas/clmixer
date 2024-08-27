"""
This module help build experiments config files and serve them to training based on a training plan

"""

import json
import os
from plugins import plugins_dict
import copy
import shutil

def generate_n_save_conditions_experiments(experiment_name,
                                           conversion_dict,
                                            representation=['Repr_Fixed','Repr_Free',"Repr_Adapted"],
                                            dataset_to_reject=[],
                                         scenarii = ['cil','indus_cil'],
                                         buffer_range = [0,10,50,100]
                                         ):
    #Create the save folder
    dirname=os.path.dirname(os.path.realpath(__file__))
    save_path=os.path.join(dirname,experiment_name)
    # Check if the directory exists before attempting to delete it
    if os.path.exists(save_path):
        # Use shutil.rmtree to delete the directory and all its contents
        shutil.rmtree(save_path)
        print(f"The directory '{save_path}' has been deleted.")
    else:
        print(f"The directory '{save_path}' does not exist.")

    os.makedirs(save_path)
    print(f"The directory '{save_path}' has been recreated.")



    default_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),"default_skeleton.json")
    base_path="/home/facto22020/Desktop/PhD/phd_datasets/curated/"

    raw_paths= [os.path.join(base_path,dataset_name) for dataset_name in  os.listdir(base_path)]
    r_paths= [os.path.join(base_path,dataset_name) for dataset_name in  dataset_to_reject]
    paths=[]

    for path in raw_paths:
        if path not in r_paths:
            paths.append(path)
        


    #Load Json skeleton
    with open(default_path, 'r') as f:
        default_dict=json.load(f)




    # Update working json for representation
    # Impact the model and data
    # free_representation_choices=["resnet18","dinov2_vits14"]
    # free_representation_backbones=["None"]

    # fixed_representation_choices=["mlp"]
    # fixed_representation_backbones=["resnet18","dinov2_vits14"]

    # adapted_representation_choices=["resnet18"]
    # adapted_representation_backbones=["None"]


    

    # conversion_dict={'Repr_Free':[(free_representation_choices,free_representation_backbones)],
    #             'Repr_Fixed':[(fixed_representation_choices,fixed_representation_backbones)],
    #             'Repr_Adapted':[(adapted_representation_choices,adapted_representation_backbones)],
    #             'Incorporation': [plugins_dict['CrossEntropyOperation']],
    #             'Retention':[None,
    #                          plugins_dict['KnowledgeDistillationOperation'],
    #                         #  plugins_dict["NCMMemoryUpdaterOperation"],
    #                          plugins_dict["RandomMemoryUpdaterOperation"],
    #                         #  [plugins_dict['KnowledgeDistillationOperation'],plugins_dict["NCMMemoryUpdaterOperation"]],
    #                           [plugins_dict['KnowledgeDistillationOperation'],plugins_dict["RandomMemoryUpdaterOperation"]]],
    #             'Bias':[None,plugins_dict['FinetuneOperation'],plugins_dict['WeightAlignOperation'],
    #              ],
    #             'Uncertainty' : [None,plugins_dict['DirichletKLLossOperation'] ]}


    configs=[]
    working_json=copy.deepcopy(default_dict)
    for scenario in scenarii:
        for path in paths:
            for given_representation in representation:
                for model in conversion_dict[given_representation][0][0]:
                    for backbone in conversion_dict[given_representation][0][1]:
                        for knowledge_incor_ in conversion_dict["Incorporation"]:
                            for ret_ in conversion_dict["Retention"]:
                                for bias_ in conversion_dict["Bias"]:
                                    for uncert_ in conversion_dict["Uncertainty"]:
                                        for buffer_size in buffer_range:
                                        
                                            working_json=copy.deepcopy(default_dict)
                                            working_json["data"]["scenario"]=scenario
                                            working_json["data"]["data_path"]=path
                                            working_json["data"]["dataset_name"]=path.split("/")[-1]
                                            working_json["model"]["model_type"]=model
                                            working_json["data"]["backbone"]=backbone

                                            if given_representation =="Repr_Adapted":
                                                #Add the frozen plugin
                                                pg = plugins_dict['FreezeNetworkBackboneOperation']
                                                working_json["plugins"].append(pg().config_template)

                                            
                                            kn_name = "None" if knowledge_incor_ is None else ''
                                            ret_name="None" if ret_ is None else "memory_"+str(buffer_size)+"_"
                                            bias_name="None" if bias_ is None else ''
                                            unc_name="None" if uncert_ is None else ''

                                            # Ensure each variable is a list if it's not already
                                            knowledge_incor = knowledge_incor_ if isinstance(knowledge_incor_, list) else [knowledge_incor_]
                                            ret = ret_ if isinstance(ret_, list) else [ret_]
                                            bias = bias_ if isinstance(bias_, list) else [bias_]
                                            uncert = uncert_ if isinstance(uncert_, list) else [uncert_]


                                            # Add config templates to working_json["plugins"] from knowledge_incor
                                            for ki in knowledge_incor:
                                                if ki is not None:
                                                    working_json["plugins"].append(ki().config_template)
                                                    # Store the name from the last added plugin's config template
                                                    kn_name += working_json["plugins"][-1]['name'] 

                                            # Add config templates to working_json["plugins"] from ret
                                            for r in ret:
                                                # if len(ret)>1:
                                                #     pass
                                                if r is not None:
                                                    config_template = r().config_template
                                                    if issubclass(r,plugins_dict["RandomMemoryUpdaterOperation"]) or issubclass(r,plugins_dict["NCMMemoryUpdaterOperation"]):
                                                        config_template["hyperparameters"]["cls_budget"] =buffer_size

                                                    working_json["plugins"].append(config_template)
                                                    # Store the name from the last added plugin's config template
                                                    ret_name += working_json["plugins"][-1]['name']+"_"

                                            # Add config templates to working_json["plugins"] from bias
                                            for b in bias:
                                                if b is not None:
                                                    working_json["plugins"].append(b().config_template)
                                                    # Store the name from the last added plugin's config template
                                                    bias_name += working_json["plugins"][-1]['name']+"_"

                                            # Add config templates to working_json["plugins"] from uncert
                                            for u in uncert:
                                                if u is not None:
                                                    working_json["plugins"].append(u().config_template)
                                                    # Store the name from the last added plugin's config template
                                                    unc_name += working_json["plugins"][-1]['name']+"_"
                                            
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
                                        


