#!/usr/bin/env bash

cd "/home/facto22020/Desktop/PhD/clmixer"
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen_cil_Adapt/kth_cil_resnet18_None_CrossEntropy_memory_10_RandomMemoryUpdater__None_None.json" -D -p -n "kth_cil_resnet18_None_CrossEntropy_memory_10_RandomMemoryUpdater__None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen_cil_Adapt/kth_cil_resnet18_None_CrossEntropy_memory_10_RandomMemoryUpdater__WeightAlign__None.json" -D -p -n "kth_cil_resnet18_None_CrossEntropy_memory_10_RandomMemoryUpdater__WeightAlign__None" --force
