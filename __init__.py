from training import ex
config_path="./config copy 3.json"
ex.add_config(config_path)
ex.add_config(
    config_path=config_path
)
r=ex.run()