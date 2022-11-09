from typing import Any
import pysd
from pysd import load
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.builders.python.python_model_builder import ModelBuilder
from pysd.translators.structures.abstract_model import (
    AbstractModel,
    AbstractSubscriptRange,
    AbstractElement,
)
from pysd.translators.structures.abstract_expressions import (
    ReferenceStructure,
    IntegStructure,
    AbstractSyntax,
    ArithmeticStructure,
    SubscriptsReferenceStructure,
    CallStructure
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import re


def to_pysd_name(name: str):
    """Convert to pysd names.

    TODO get the real equivalent from pysd
    """
    s = name.lower()
    clean_s = s.replace(" ", "_")

    # Make spaces into underscores
    s = re.sub(r"[\s\t\n_]+", "_", s)
    return s


def add_subscript_to_abstact_syntax(
    ast: AbstractSyntax | int | float | np.ndarray,
    N_MEMBERS: int,
    initial: np.ndarray | None = None,
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
        ast.flow = add_subscript_to_abstact_syntax(
            ast.flow, N_MEMBERS, initial=initial
        )
        ast.initial = add_subscript_to_abstact_syntax(
            ast.initial, N_MEMBERS, initial=initial
        )
        print(ast)
        return ast
    elif isinstance(ast, np.ndarray):
        shape = np.shape(ast)
        return np.tile(ast, N_MEMBERS).reshape((N_MEMBERS,) + shape)
    elif isinstance(ast, int | float):
        # return np.full(N_MEMBERS, ast)
        if initial is None:
            raise ValueError(f"{ast=} requires initial conditions.")
        return initial
    elif isinstance(ast, ArithmeticStructure):
        ast.arguments = tuple(
            [
                add_subscript_to_abstact_syntax(
                    arg, N_MEMBERS, initial=initial
                )
                for arg in ast.arguments
            ]
        )
        return ast

    else:
        raise TypeError("Cannot add subscript to ast:", ast, type(ast))

    raise RuntimeError("should not arrive here")


def add_merge_operation_to_abstact_syntax(
    ast: AbstractSyntax | int | float | np.ndarray,
    merge_dict: dict[str, str],
) -> AbstractSyntax:
    if isinstance(ast, ReferenceStructure):
        if not ast.reference in merge_dict.keys():
            return ast
        # apply the merging operation to the ast
        return CallStructure(merge_dict[ast.reference], (ast))
    elif isinstance(ast, ArithmeticStructure):
        ast.arguments = tuple([
            # apply it recursively to the arguments
            add_merge_operation_to_abstact_syntax(arg_ast, merge_dict) for arg_ast in ast.arguments
        ])

        return ast
    else:
        raise NotImplementedError(
            "Cannot add merge operation to ast:", type(ast), ast, )

    raise RuntimeError("should not arrive here")


def get_forward_dependencies(model: pysd.py_backend.model.Model) -> dict[str, list[str]]:
    """Get a dictionary mapping for each variable which one depend on it."""
    forward_deps: dict[str, list[str]] = {}
    for dep_var, dep_dict in model.dependencies.items():
        for depending_var, depending_value in dep_dict.items():
            if depending_var not in forward_deps:
                forward_deps[depending_var] = []
            forward_deps[depending_var].append(dep_var)
    return forward_deps


def add_subscript(
    model: AbstractModel,
    subscript_list: list[str],
    subscripted_vars_dict: dict[str, dict[str, np.ndarray | dict[str, str]]],
    subscript_dim_name="Ensemble Dimension",
):
    """Add a subscript to some variables of the model.



    Args:
        model (AbstractModel): The model that has its variables modified
        subscript_list (list[str]): The name of the subsrcipts dimensions
        subscripted_vars_dict: A dictonary that tells how the subscripts
            should be added.
    """
    # Create the subscript range

    sub_range = AbstractSubscriptRange(
        name=subscript_dim_name,
        subscripts=subscript_list,
        mapping=[],
    )

    merg_on_variables = {
        merge_var: {to_pysd_name(var_name): operation} for var_name, d in subscripted_vars_dict.items()
        for merge_var, operation in d.get("merge", {}).items()
    }
    print("variables for merging", merg_on_variables)
    # TODO: maybe we can just add an arithmetic structure that does the
    # average or so
    for section in model.sections:
        section.subscripts.append(sub_range)
        section_elementsnames = [el.name for el in section.elements]
        print(section_elementsnames)
        for el in section.elements:
            el: AbstractElement
            print(el.name)
            if el.name in subscripted_vars_dict.keys():
                # check dependencies are subscripted or in splitter variables
                subscript_dict = subscripted_vars_dict[el.name]
                for c in el.components:
                    #print("before", c)
                    c.subscripts[0].insert(0, subscript_dim_name)
                    c.ast = add_subscript_to_abstact_syntax(
                        c.ast, len(subscript_list), initial=subscript_dict.get('initial', None)
                    )
                    #print("after", c)
            if el.name in merg_on_variables.keys():
                print("Handling merging",
                      merg_on_variables[el.name], "for", repr(el))
                for c in el.components:
                    print("before", repr(c))
                    c.ast = add_merge_operation_to_abstact_syntax(
                        c.ast, merg_on_variables[el.name]
                    )
                    print("after", repr(c))
