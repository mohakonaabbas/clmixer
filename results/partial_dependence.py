import pandas as pd
import plot_api
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from time import time
df1 = plot_api.getAllValidExperiments(databaseName="representation_free")
preprocessing=plot_api.prepare_inspection_pdp(df1,root_path="/home/mohamedphd/Documents/phd/Datasets/curated/")

X=preprocessing["x"]
y=preprocessing["y"]
features_info=preprocessing["features_info"]
common_params=preprocessing["common_params"]
model=preprocessing["models"]


# Create the dependance plot dataset
print("Computing partial dependence plots...")

_, ax = plt.subplots(figsize=(9, 8))


display = PartialDependenceDisplay.from_estimator(
    model["gradient_boosting"][1],
    X,
    **features_info,
    ax=ax,
    **common_params,
)

pd_results=display.pd_results
_ = display.figure_.suptitle(
    (
        "Partial dependence of the IL parameters"
    ),
    fontsize=16,
)

plt.show()

plt.show()