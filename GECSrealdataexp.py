# Importing Packages
import numpy as np
import pandas as pd
from ecDAGlearn import gecs

# Red Wine quality data:

#import data set
wine_data = pd.read_csv('wine+quality/winequality-red.csv', delimiter=';')


#convert to np array and store column indices:
wine_variables =wine_data.columns[:-1]
wine_data_np = wine_data.to_numpy()
wine_data_np = wine_data_np[:, :-1]

#run GECS
wineGECSgraph, wineGECScoloring = gecs(wine_data_np)

#store results
np.savez(
        "r_wine_GECS_results.npz",
        wineGECSgraph=wineGECSgraph,
        wineGECScoloring=wineGECScoloring,
        winevariables=wine_variables
)


# White Wine quality data:

#import data set
w_wine_data = pd.read_csv('wine+quality/winequality-white.csv', delimiter=';')


#convert to np array and store column indices:
w_wine_variables = w_wine_data.columns[:-1]
w_wine_data_np = w_wine_data.to_numpy()
w_wine_data_np = w_wine_data_np[:, :-1]

#run GECS
w_wineGECSgraph, w_wineGECScoloring = gecs(w_wine_data_np)

#store results
np.savez(
        "w_wine_GECS_results.npz",
        w_wineGECSgraph=w_wineGECSgraph,
        w_wineGECScoloring=w_wineGECScoloring,
        w_winevariables=w_wine_variables
)



# Sachs observational data:

#import observational data set
sachs_obs_data = pd.read_csv('sachs_obs_data.csv')


#convert to np array and store column indices:
sachs_variables =sachs_obs_data.columns
sachs_obs_data_np = sachs_obs_data.to_numpy()

#run GECS
sachsGECSgraph, sachsGECScoloring = gecs(sachs_obs_data_np)

#store results
np.savez(
        "sachs_obs_GECS_results.npz",
        sachsGECSgraph=sachsGECSgraph,
        sachsGECScoloring=sachsGECScoloring,
        sachsvariables=sachs_variables
)