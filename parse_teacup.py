# %%
import pysd
from pysd import load
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.builders.python.python_model_builder import ModelBuilder
from pysd.builders.python.python_model_builder_subscripts import SubscripterModelBuilder
from pysd.translators.structures.abstract_model import (
    AbstractModel,
    AbstractSubscriptRange,
)
from pysd.translators.structures.abstract_expressions import ReferenceStructure
import numpy as np

# %%
# mdl_file = "teacup.mdl"
mdl_file = "test_subscript_3d_arrays_lengthwise.mdl"
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


# %%
N_MEMBERS = 5
vars = ["Three Dimensional Constant", "Three Dimensional Variable"]

ven_file.parse()
# get AbstractModel
abs_model = ven_file.get_abstract_model()


def add_subscript(model: AbstractModel):
    # Create the subscript range
    sub_range = AbstractSubscriptRange(
        name="Ensemble Dimension",
        subscripts=[f"m_{i}" for i in range(N_MEMBERS)],
        mapping=[],
    )
    for section in model.sections:
        section.subscripts.append(sub_range)

        for el in section.elements:
            print(el.name)
            if el.name not in vars:
                continue
            for c in el.components:
                print(c)
                print(c.ast)
                c.subscripts[0].insert(0, "Ensemble Dimension")
                print(c.subscripts)
                if isinstance(c.ast, np.ndarray):
                    shape = np.shape(c.ast)
                    c.ast = np.tile(c.ast, N_MEMBERS).reshape((N_MEMBERS,) + shape)
                elif isinstance(c.ast, ReferenceStructure):
                    print(c.ast)
                    c.ast.subscripts.subscripts.insert(0, "Ensemble Dimension")
                    print(c.ast)
                else:
                    raise TypeError(c.ast, type(c.ast))


add_subscript(abs_model)
# %%
py_model_modified = SubscripterModelBuilder(abs_model)
# change the name of the file
py_model_modified.sections[0].path = py_model_modified.sections[0].path.with_stem(
    "modified"
)
f_name = py_model_modified.build_model()
m = load(f_name)
m.run()
# %%
# build Python file
py_model_file = ModelBuilder(abs_model).build_model()


# %%
# load Python file
model = load(py_model_file, data_files, initialize, missing_values)
model.mdl_file = str(mdl_file)
