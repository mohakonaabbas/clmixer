
from configs import generate_save_conditions_experiments,generate_sh_script,generate_save_conditions_experiments_free
dirpath_fixed=generate_save_conditions_experiments(experiment_name="IL_Experiments_Frozen",
                                             representation=['Repr_Fixed'],scenarii=['indus_cil'])
# dirpath_free=generate_save_conditions_experiments_free(experiment_name="IL_Experiments_Repr_Free",
#                                              representation=['Repr_Free'])
generate_sh_script(path=dirpath_fixed,savingDatabase="Frozen",base_path= "\"/home/facto22020/Desktop/PhD/clmixer\"")
# generate_sh_script(path=dirpath_free,savingDatabase="representation_free")