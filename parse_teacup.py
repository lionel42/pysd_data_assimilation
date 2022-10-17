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
    SubscriptsReferenceStructure,
)
import numpy as np
import pandas as pd

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
perturb_sigma = 0.2
N_MEMBERS = 5
subscript_list = [f"m_{i}" for i in range(N_MEMBERS)]
subscript_tuple = tuple(subscript_list)
vars = ["Characteristic Time", "Teacup Temperature"]

ven_file.parse()
# get AbstractModel
abs_model = ven_file.get_abstract_model()


def add_subscript_to_abstact_syntax(
    ast: AbstractSyntax | int | float | np.ndarray,
) -> AbstractSyntax:
    """Return a new ast with the added substcript."""
    if isinstance(ast, ReferenceStructure):
        if ast.subscripts is not None:
            ast.subscripts.subscripts.insert(0, "Ensemble Dimension")
        else:
            # This will bug later if you use tuple instead of list ;(
            # Actually it also raised a warning, but I guess it comes
            # from the IntegStructure flow thing that duplicates because
            # we already set values to the flow
            # ast.subscripts = SubscriptsReferenceStructure(subscript_list)
            pass
        return ast
    elif isinstance(ast, IntegStructure):
        print(ast)
        ast.flow = add_subscript_to_abstact_syntax(ast.flow)
        ast.initial = add_subscript_to_abstact_syntax(ast.initial)
        print(ast)
        return ast
    elif isinstance(ast, np.ndarray):
        shape = np.shape(ast)
        return np.tile(ast, N_MEMBERS).reshape((N_MEMBERS,) + shape)
    elif isinstance(ast, int | float):
        # return np.full(N_MEMBERS, ast)
        return ast + np.random.normal(scale=perturb_sigma, size=N_MEMBERS) * ast
    elif isinstance(ast, ArithmeticStructure):
        ast.arguments = tuple(
            [add_subscript_to_abstact_syntax(arg) for arg in ast.arguments]
        )
        return ast

    else:
        raise TypeError("Cannot add subscript to ast:", ast, type(ast))

    raise RuntimeError("should not arrive here")


def add_subscript(model: AbstractModel):
    # Create the subscript range

    sub_range = AbstractSubscriptRange(
        name="Ensemble Dimension",
        subscripts=subscript_list,
        mapping=[],
    )
    for section in model.sections:
        section.subscripts.append(sub_range)

        for el in section.elements:
            print(el.name)
            if el.name not in vars:
                continue
            for c in el.components:
                print("before", c)
                c.subscripts[0].insert(0, "Ensemble Dimension")
                c.ast = add_subscript_to_abstact_syntax(c.ast)
                print("after", c)


add_subscript(abs_model)
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
A = X - E_X.reshape(-1, 1)
C = A.T.dot(A) / (N_MEMBERS - 1)

R = 0.08

X, E_X, A, C
#%% run the model
df_out = m.run()
df_out
# %%
# build Python file
py_model_file = ModelBuilder(abs_model).build_model()


# %%
# load Python file
model = load(py_model_file, data_files, initialize, missing_values)
model.mdl_file = str(mdl_file)

# %% specify the observations
SIGMA = 0.1
obs_vars = ['Teacup Temperature']
observations = pd.DataFrame(
    {
        "time": df_out.index.to_numpy(),
        "obs": df_out["Teacup Temperature[m_0]"].to_numpy()
        + np.random.normal(0, SIGMA, len(df_out))
        * df_out["Teacup Temperature[m_0]"].to_numpy(),
        "obs_err": np.random.normal(0, SIGMA, len(df_out))
        * df_out["Teacup Temperature[m_0]"].to_numpy(),
    }
)
# %% extract data as kahlman notation

D = np.array(
    [
        observations["obs"] + np.random.normal(0, R * observations["obs"])
        for _ in range(N_MEMBERS)
    ]
).T
D
# %% extract model run from observation
H_X =
# %% find the posterior
X_P = X + C
