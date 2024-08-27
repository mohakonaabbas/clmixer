from training import ex
config_path="kth_cil_mlp_dinov2vits14_CrossEntropy_None_None_None.json"
ex.add_config(config_path)
ex.add_config(config_path=config_path)
r=ex.run()