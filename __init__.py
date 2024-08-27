from training import ex
<<<<<<< HEAD
config_path="kth_cil_mlp_dinov2vits14_CrossEntropy_None_None_None.json"
=======
config_path="./modif_ce.json"
>>>>>>> d4f5009aa8d94fcd3fd07973c97869e0c1f122fd
ex.add_config(config_path)
ex.add_config(config_path=config_path)
r=ex.run()