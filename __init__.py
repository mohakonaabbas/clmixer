from training import ex
from configs import generate_save_conditions_experiments
configs=generate_save_conditions_experiments(experiment_name="Conditions_Of_IL_Experiments_Repr_Fixed",representation=['Repr_Fixed'])
config_path="./configs/default_skeleton copy.json"
ex.add_config(config_path)
for config in configs:
    r=ex.run(config_updates=config)