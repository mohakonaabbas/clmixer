from training import ex
config_path="./test_config.json"
ex.add_config(config_path)
ex.add_config(config_path=config_path)
r=ex.run()