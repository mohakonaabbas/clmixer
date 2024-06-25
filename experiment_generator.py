
from configs import generate_save_conditions_experiments,generate_sh_script,generate_save_conditions_experiments_free, generate_n_save_conditions_experiments
from plugins import plugins_dict

free_representation_choices=["resnet18"]
free_representation_backbones=["None"]

fixed_representation_choices=["mlp"]
fixed_representation_backbones=["resnet18","dinov2_vits14"]

adapted_representation_choices=["resnet18"]
adapted_representation_backbones=["None"]

# conversion_dict={'Repr_Free':[(free_representation_choices,free_representation_backbones)],
#             'Repr_Fixed':[(fixed_representation_choices,fixed_representation_backbones)],
#             'Repr_Adapted':[(adapted_representation_choices,adapted_representation_backbones)],
#             'Incorporation': [plugins_dict['CrossEntropyOperation']],
#             'Retention':[None,
#                             plugins_dict['KnowledgeDistillationOperation'],
#                         #  plugins_dict["NCMMemoryUpdaterOperation"],
#                             plugins_dict["RandomMemoryUpdaterOperation"],
#                         #  [plugins_dict['KnowledgeDistillationOperation'],plugins_dict["NCMMemoryUpdaterOperation"]],
#                             [plugins_dict['KnowledgeDistillationOperation'],plugins_dict["RandomMemoryUpdaterOperation"]]],
#             'Bias':[None,plugins_dict['FinetuneOperation'],plugins_dict['WeightAlignOperation'],
#                 ],
#             'Uncertainty' : [None,plugins_dict['DirichletKLLossOperation'] ]}

conversion_dict={'Repr_Free':[(free_representation_choices,free_representation_backbones)],
            'Repr_Fixed':[(fixed_representation_choices,fixed_representation_backbones)],
            'Repr_Adapted':[(adapted_representation_choices,adapted_representation_backbones)],
            'Incorporation': [plugins_dict['CrossEntropyOperation']],
            'Retention':[plugins_dict["RandomMemoryUpdaterOperation"]],
            'Bias':[None,plugins_dict['FinetuneOperation'],plugins_dict['WeightAlignOperation']],
            'Uncertainty' : [None]}
exp_name = "IL_Experiments_Frozen_cil_Adapt"
dirpath_fixed=generate_n_save_conditions_experiments(experiment_name=exp_name,
                                                     conversion_dict= conversion_dict,
                                             representation=['Repr_Free'],
                                             dataset_to_reject=["dagm","nouvel_op","mvtec","magnetic"],
                                             buffer_range=[10],
                                             scenarii=['cil'])
# dirpath_fixed=generate_save_conditions_experiments(experiment_name=exp_name,
#                                              representation=['Repr_Fixed'],scenarii=['cil'])
# dirpath_free=generate_save_conditions_experiments_free(experiment_name="IL_Experiments_Repr_Free",
#                                              representation=['Repr_Free'])
generate_sh_script(path=dirpath_fixed,savingDatabase="Frozen",base_path= "\"/home/facto22020/Desktop/PhD/clmixer\"")
# generate_sh_script(path=dirpath_free,savingDatabase="representation_free")