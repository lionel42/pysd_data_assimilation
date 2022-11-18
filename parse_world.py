# %%
from typing import Any
import pysd
from pysd import load
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.builders.python.python_model_builder import ModelBuilder
from pysd.translators.structures.abstract_model import (
    AbstractModel,
    AbstractSubscriptRange,
)
from pysd.translators.structures.abstract_expressions import (
    ReferenceStructure,
    IntegStructure,
    AbstractSyntax,
    ArithmeticStructure,
    CallStructure,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import logging
from utils import add_subscript, get_all_dependencies, get_forward_dependencies

logging.basicConfig()
# %%
mdl_file = "world3.mdl"
# mdl_file = "test_subscript_3d_arrays_lengthwise.mdl"
# pysd.read_vensim(mdl_file)

# %%
data_files = None
initialize = True
missing_values = "warning"
split_views = False
encoding = None

# %%
# Read and parse Vensim file
ven_file = VensimFile(mdl_file, encoding=encoding)

# %%
ven_file.parse()

if split_views:
    # split variables per views
    subview_sep = kwargs.get("subview_sep", "")
    ven_file.parse_sketch(subview_sep)


# %%
N_MEMBERS = 3
subscript_list = [f"m_{i}" for i in range(N_MEMBERS)]

subscript_tuple = tuple(subscript_list)
# TODO: add variables that are initial conditions

ven_file.parse()
# get AbstractModel
abs_model = ven_file.get_abstract_model()
py_model_original = load(ModelBuilder(abs_model).build_model())

#%% look for all variables

py_model_original.namespace.keys()
#%% create variables

#var = 'initial arable land'
var = 'Arable Land'
# First we can look at the dependencies of our model
get_all_dependencies(py_model_original, var)

#%%
#%load_ext autoreload
#%autoreload 2
from utils import add_subscript
# Dictonary defining how variables are behaving with the addition
# of subscript.
logging.getLogger('add_merge_operation_to_abstact_syntax').setLevel(level=logging.DEBUG)
logging.getLogger('add_split_operation_to_abstact_syntax').setLevel(level=logging.DEBUG)
logging.getLogger('add_subscript').setLevel(level=logging.DEBUG)

# TODO: add option to propagate the subscript.
subscripted_vars_dict = {
    # Arable Land' will be subscripted
    "Arable Land": {
        # Define the initial values of the subscripts
        "initial": np.random.normal(80, 10, N_MEMBERS),
        # Varialbes that should merge subscripted variables into one before using (sum or mean)
        # This is very important as variables could be propagating the subscript to
        # others variables generating unexpected behaviours
        "merge": {
            # we specify that we want to use the mean of all
            # the teapcups temperatures as input for the
            # heat loss to room function
            'arable_land_in_gigahectares_gha': 'mean',
            'food': 'mean',
            'land_fr_cult': 'mean',
            'agricultural_input_per_hectare': 'mean',
            'land_erosion_rate': 'mean',
            'potential_jobs_agricultural_sector': 'mean',
            'persistent_pollution_generation_agriculture': 'mean',
        },
        # Variables splitted into a subscript before they are used by this one
        # If you don't put anything here it is in genral okay, as the float
        # value will be used for all the subscripts
        "split": {
            "initial_arable_land": {
                "factors": np.random.normal(1, 0.1, N_MEMBERS)
            }
        },
    }
}
add_subscript(abs_model, subscript_list, subscripted_vars_dict, )


# %%
py_model_modified = ModelBuilder(abs_model)
# change the name of the file
py_model_modified.sections[0].path = py_model_modified.sections[0].path.with_stem(
    "modified"
)
f_name = py_model_modified.build_model()
m = load(f_name)
#%% build X
# https://en.wikipedia.org/wiki/Ensemble_Kalman_filter#Ensemble_Kalman_Filter
vars
X = np.concatenate([[getattr(m.components, m.namespace[f"{var}"])() for var in vars]])
E_X = np.mean(X, axis=1)
A = X - E_X.reshape(-1, 1)  # Deviations from the state
C = A.T.dot(A) / (N_MEMBERS - 1)

# Standard deviation of th eobservations
SIGMA = 0.1


X, E_X, A, C
#%% run the model
df_out = m.run()
df_out


# %%


obs_vars = ["Teacup Temperature"]
observations = pd.DataFrame(
    {
        "time": df_out.index.to_numpy(),
        "obs": df_out["Teacup Temperature[m_0]"].to_numpy()
        + np.random.normal(0, SIGMA, len(df_out))
        * df_out["Teacup Temperature[m_0]"].to_numpy(),
        "obs_err": SIGMA * df_out["Teacup Temperature[m_0]"].to_numpy(),
        "variable": "Teacup Temperature",
    }
).iloc[::5]

observations = pd.DataFrame(
    {
        "time": [1, 3, 20, 28],
        "obs": [160, 120, 105, 100],
        "obs_err": 20,
        "variable": "Teacup Temperature",
    }
)
# %% extract data as kahlman notation

D = np.array(
    [
        np.random.normal(observations["obs"], observations["obs_err"])
        for _ in range(N_MEMBERS)
    ]
).T
D
# covariance of the observations
D_ = D - observations["obs"].to_numpy().reshape((-1, 1)) / N_MEMBERS
R = (D_ @ D_.T) / (N_MEMBERS - 1)
# I cannot make it work with the above, but with eye it works magic
R = np.eye(len(observations))


# %% extract model run from observation
# n_obs x n_members
HX = np.array(
    [
        df_out.lookup(observations["time"], observations["variable"] + f"[m_{i}]")
        for i in range(N_MEMBERS)
    ]
).T


# %%
# Observation matrix-free implementation (wiki)

HA = HX - HX.mean(axis=1).reshape(-1, 1)

# n_obs x n_obs
P = HA @ HA.T / (N_MEMBERS - 1) + R

# Caluclate the kahlman gain
# nstates x n_obs
K = A @ HA.T @ np.linalg.inv(P) / (N_MEMBERS - 1)


#%% Find the posterior distribution
X_POST = X + K @ (D - HX)
X_POST
# %%
for m in range(N_MEMBERS):
    plt.plot(df_out.index, df_out[f"Teacup Temperature[m_{m}]"], color="grey")

obs_col = "blue"
plt.plot(observations["time"], observations["obs"], color=obs_col)
plt.fill_between(
    observations["time"],
    observations["obs"],
    observations["obs"] + observations["obs_err"],
    color=obs_col,
)
plt.fill_between(
    observations["time"],
    observations["obs"],
    observations["obs"] - observations["obs_err"],
    color=obs_col,
)
# %% Run the posterior, this will work only for 1d ensemble at the moment
# load Python file
m = load(f_name)
posterior_vars_dict = {
    var_name: xr.DataArray(X_POST[i, :], *m.get_coords(var_name))
    for i, var_name in enumerate(vars)
}
df_post = m.run(
    params={"Characteristic Time": posterior_vars_dict["Characteristic Time"]},
    initial_condition=(
        0,
        {"Teacup Temperature": posterior_vars_dict["Teacup Temperature"]},
    ),
)

# %% plot posterio
for m in range(N_MEMBERS):
    plt.plot(df_post.index, df_post[f"Teacup Temperature[m_{m}]"], color="grey")

obs_col = "blue"
plt.plot(observations["time"], observations["obs"], color=obs_col)
plt.fill_between(
    observations["time"],
    observations["obs"],
    observations["obs"] + observations["obs_err"],
    color=obs_col,
)
plt.fill_between(
    observations["time"],
    observations["obs"],
    observations["obs"] - observations["obs_err"],
    color=obs_col,
)

# %%
