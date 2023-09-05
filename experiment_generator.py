
from configs import generate_save_conditions_experiments,generate_sh_script,generate_save_conditions_experiments_free
dirpath_fixed=generate_save_conditions_experiments(experiment_name="IL_Experiments_Repr_Fixed",
                                             representation=['Repr_Fixed'])
dirpath_free=generate_save_conditions_experiments_free(experiment_name="IL_Experiments_Repr_Free",
                                             representation=['Repr_Free'])
generate_sh_script(path=dirpath_fixed,savingDatabase="experiments_representations")
generate_sh_script(path=dirpath_free,savingDatabase="experiments_representations")