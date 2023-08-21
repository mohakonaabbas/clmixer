
from configs import generate_save_conditions_experiments,generate_sh_script
dirpath=generate_save_conditions_experiments(experiment_name="IL_Experiments_Repr_Fixed",
                                             representation=['Repr_Fixed'])
generate_sh_script(path=dirpath)