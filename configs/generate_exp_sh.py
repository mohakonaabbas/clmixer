"""
This module buil sh files to run experiments in a queue

"""

import os



def generate_sh_script(path):
    files=["#!/usr/bin/env bash\n\n","cd \"/home/mohamedphd/Documents/phd/clmixer\"\n"]
    savepath=path+'.sh'
    config_files=os.listdir(path)
    for config_file in config_files:
        abspath=os.path.abspath(os.path.join(path,config_file))
        command=f"python3 training.py with \"{abspath}\" -D -p -n \"{config_file[:-5]}\" --force\n"
        files.append(command)
    
    with open(savepath,'w') as f:
        f.writelines(files)

if __name__=="__main__":
    generate_sh_script("/home/mohamedphd/Documents/phd/clmixer/configs/Conditions_Of_IL_Experiments_Repr_Fixed")
