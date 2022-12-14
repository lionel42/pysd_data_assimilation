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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from utils import add_subscript


# %%
mdl_file = "teacup.mdl"
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
import importlib
import utils
importlib.reload(utils)
import utils
from utils import add_subscript, get_forward_dependencies
N_MEMBERS = 3
subscript_list = [f"m_{i}" for i in range(N_MEMBERS)]

subscript_tuple = tuple(subscript_list)
vars = ["Characteristic Time", "Teacup Temperature"]
# TODO: add variables that are initial conditions

ven_file.parse()
# get AbstractModel
abs_model = ven_file.get_abstract_model()
original_model = ModelBuilder(abs_model)
f_name = original_model.build_model()
m = load(f_name)

deps = get_forward_dependencies(m)

# Dictonary defining how variables are behaving with the addition
# of subscript.
subscripted_vars_dict = {
    # Teacup temperature will be subscripted
    "Teacup Temperature": {
        # Define the initial values of the subscripts
        "initial": np.random.normal(80, 10, N_MEMBERS),
        # Varialbes that should merge subscripted variables into one before using (sum or mean)
        "merge": {
            # heat loss reqired teacup temperature
            # by specifying a merge on the teacup temperature
            # we specify that we want to use the mean of all
            # the teapcups temperatures as input for the
            # heat loss to room function
            "Heat Loss to Room": 'mean'
        },
        # Variables splitted into a subscript before they are used by this one
        "split": {},
    },
    "Characteristic Time": {
        "initial": np.random.normal(10, 2, N_MEMBERS),
        "merge": {"Heat Loss to Room": 'mean'}, # Variables that should merge subscripted variables into one before using (sum or mean)
        "split": {}, # Variables that should be split before they are used by this
    }
}


add_subscript(abs_model, subscript_list,subscripted_vars_dict)
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
