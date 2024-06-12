#!/usr/bin/env bash

cd "/home/facto22020/Desktop/PhD/clmixer"
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None.json" -D -p -n "kth_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_None_None_None.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/kth_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None.json" -D -p -n "kth_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None.json" -D -p -n "nouvelop_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_None_None_None.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/nouvelop_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None.json" -D -p -n "nouvelop_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None.json" -D -p -n "mvtec_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_None_None_None.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/mvtec_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None.json" -D -p -n "mvtec_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None.json" -D -p -n "dagm_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_None_None_None.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/dagm_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None.json" -D -p -n "dagm_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None.json" -D -p -n "magnetic_induscil_mlp_dinov2vits14_CrossEntropy_None_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_KnowledgeDistillation_WeightAlign_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_None_Finetune_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_None_Finetune_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_None_None_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_None_None_None.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_None_None_None" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_DirichletKLLoss" --force
python3 training.py with "/home/facto22020/Desktop/PhD/clmixer/configs/IL_Experiments_Frozen/magnetic_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None.json" -D -p -n "magnetic_induscil_mlp_resnet18_CrossEntropy_None_WeightAlign_None" --force
