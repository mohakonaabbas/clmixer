#if you use Google Colab or have wget, you can (but do not have to) load the dataset with

reprensation=['SSL Representation','Perfect Fixed','Free']
knowledge_incorporation=['CE','L2','Dirichlet CE']
knowledge_retention=["None","KD",'Dynamic Arch',"big buffer"]
Bias_Mitigation=["None","wa","bic","finetuning"]
uncertainty_reduction=["None","dirichlet","conformal"]

results={"representation_learning":[],"knowledge_incorporation":[],"knowledge_retention":[],"bias_Mitigation":[],"uncertainty_reduction":[],}

for rep in reprensation:
    for ret in knowledge_retention:
        for inc in knowledge_incorporation:
            for bias in Bias_Mitigation:
                for uncert in uncertainty_reduction:
    
                    results["bias_Mitigation"].append(bias)
                    results['knowledge_incorporation'].append(inc)
                    results["knowledge_retention"].append(ret)
                    results["representation_learning"].append(rep)
                    results["uncertainty_reduction"].append(uncert)

import pandas as pd
(pd.DataFrame.from_dict(results)).to_csv('./exp_plan.csv')
