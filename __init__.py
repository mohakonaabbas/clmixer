from training import ex
config_path="./modif_ce.json"
ex.add_config(config_path)
ex.add_config(config_path=config_path)
r=ex.run()