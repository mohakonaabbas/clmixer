from training import ex
config_path="./config copy.json"
ex.add_config(config_path)
ex.add_config(
    config_path=config_path
)
r=ex.run()