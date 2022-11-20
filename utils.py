import logging
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
    CallStructure,
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

    # Remove invalid characters
    s = re.sub(r"[^0-9a-zA-Z_]", "", s)
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
        ast.flow = add_subscript_to_abstact_syntax(ast.flow, N_MEMBERS, initial=initial)
        ast.initial = add_subscript_to_abstact_syntax(
            ast.initial, N_MEMBERS, initial=initial
        )
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
                add_subscript_to_abstact_syntax(arg, N_MEMBERS, initial=initial)
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
    subscript_dim_name: str,
) -> AbstractSyntax:
    logger = logging.getLogger("add_merge_operation_to_abstact_syntax")
    logger.debug(f"{ast=}, {merge_dict=}")
    if isinstance(ast, ReferenceStructure):
        ast_var = to_pysd_name(ast.reference)
        if not ast_var in merge_dict.keys():
            # This is a variable that does not have to be merged
            return ast
        merge_method = merge_dict[ast_var]
        accepted_operations = ["sum", "prod", "mean"]
        if merge_method in accepted_operations:
            # apply the merging operation on the subscript axis to the ast
            call_func = ReferenceStructure(
                merge_method,
                subscripts=SubscriptsReferenceStructure((subscript_dim_name + "!",)),
            )
            ast.subscripts = SubscriptsReferenceStructure((subscript_dim_name + "!",))
            return CallStructure(call_func, (ast,))
        elif (
            merge_method.startswith("weighted_")
            and merge_method.split("_")[1] in accepted_operations
        ):
            call_func = ReferenceStructure(
                merge_method,
                subscripts=SubscriptsReferenceStructure((subscript_dim_name + "!",)),
            )
            ast.subscripts = SubscriptsReferenceStructure((subscript_dim_name + "!",))
            return CallStructure(call_func, (ast,))
        else:
            raise ValueError(f"Unknown merging method {merge_method}")
    elif isinstance(ast, ArithmeticStructure):
        ast.arguments = tuple(
            [
                # apply it recursively to the arguments that are variables (skip int, floats)
                arg_ast
                if isinstance(arg_ast, (int, float))
                else add_merge_operation_to_abstact_syntax(
                    arg_ast, merge_dict, subscript_dim_name
                )
                for arg_ast in ast.arguments
            ]
        )

        return ast
    else:
        raise NotImplementedError(
            "Cannot add merge operation to ast:",
            type(ast),
            ast,
        )

    raise RuntimeError("should not arrive here")


def add_split_operation_to_abstact_syntax(
    ast: AbstractSyntax | int | float | np.ndarray,
    # Maps variable to split operations
    split_dict: dict[str, str|dict[str, list]],
    subscript_dim_name: str,
) -> AbstractSyntax:
    logger = logging.getLogger("add_split_operation_to_abstact_syntax")
    logger.debug(f"{ast=}, {split_dict=}")
    if isinstance(ast, ReferenceStructure):
        ast_var = to_pysd_name(ast.reference)
        if not ast_var in split_dict.keys():
            # This is a variable that does not have to be splitted
            return ast
        split_method = split_dict[ast_var]
        if isinstance(split_method, dict):
            if "factors" in split_method:
                # multiply the variable to the factors
                return ArithmeticStructure(["*"], [ast, split_method['factors']])
            else:
                raise ValueError(f"Unknown splitting method {split_method}")
        else:
            raise ValueError(f"Unknown splitting method {split_method}")
    elif isinstance(ast, IntegStructure):
        # Also add the variables to the splitted asts
        ast.flow = add_split_operation_to_abstact_syntax(ast.flow, split_dict, subscript_dim_name)
        ast.initial = add_split_operation_to_abstact_syntax(ast.initial, split_dict, subscript_dim_name)
        return ast

    elif isinstance(ast, ArithmeticStructure):
        ast.arguments = tuple(
            [
                # apply it recursively to the arguments that are variables (skip int, floats)
                arg_ast
                if isinstance(arg_ast, (int, float))
                else add_split_operation_to_abstact_syntax(
                    arg_ast, split_dict, subscript_dim_name
                )
                for arg_ast in ast.arguments
            ]
        )

        return ast
    else:
        raise NotImplementedError(
            "Cannot add split operation to ast:",
            type(ast),
            ast,
        )

    raise RuntimeError("should not arrive here")

def get_forward_dependencies(
    model: pysd.py_backend.model.Model,
) -> dict[str, list[str]]:
    """Get a dictionary mapping for each variable which one depend on it."""
    # TODO implement reading diraclty the get_depencenicies from pysd
    forward_deps: dict[str, list[str]] = {}
    for dep_var, dep_dict in model.dependencies.items():
        dep_dict: dict[str, int | str]
        for depending_var, depending_value in dep_dict.items():
            if depending_var not in forward_deps:
                forward_deps[depending_var] = []
            forward_deps[depending_var].append(dep_var)
            if depending_var.startswith("_"):
                # intermediate variables (integ, lookuptables, ...)
                # We want the original variable of the model
                if depending_var.startswith("_integ_"):
                    original_var = depending_var[7:]
                    for integ_dep_dicts in model.dependencies[depending_var].values():
                        for dep_var_integ in integ_dep_dicts.keys():
                            if dep_var_integ not in forward_deps:
                                forward_deps[dep_var_integ] = []
                            # Add all the dependencies of the integ in it
                            forward_deps[dep_var_integ].append(original_var)

    return forward_deps


def get_all_dependencies(model: pysd.py_backend.model.Model, var: str):
    """Get prior and forward dependencies for a variables."""
    vars = [var]
    prior_deps = model.get_dependencies(vars)
    # TODO: make something less expensive if possible
    py_name = model.namespace[var] if var in model.namespace else var
    forward_deps = get_forward_dependencies(model)[py_name]
    return prior_deps, forward_deps


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
    logger = logging.getLogger("add_subscript")
    # Create the subscript range
    sub_range = AbstractSubscriptRange(
        name=subscript_dim_name,
        subscripts=subscript_list,
        mapping=[],
    )

    subscripted_vars_dict = {
        to_pysd_name(key): item for key, item in subscripted_vars_dict.items()
    }
    logger.debug(f"{subscripted_vars_dict=}")

    # Extract variables that will be used for merging
    merg_on_variables = {}
    for var_name, d in subscripted_vars_dict.items():
        for merge_var, operation in d.get("merge", {}).items():
            merge_var = to_pysd_name(merge_var)
            if merge_var not in merg_on_variables:
                merg_on_variables[merge_var] = {}
            merg_on_variables[merge_var][var_name] = operation

    # Extract variables that will be used for splitting (similar to merge)
    split_on_variables = {}
    for var_name, d in subscripted_vars_dict.items():
        for split_var, operation in d.get("split", {}).items():
            split_var = to_pysd_name(split_var)
            if var_name not in split_on_variables:
                split_on_variables[var_name] = {}
            # Map to each variable which input should be split using which operation
            split_on_variables[var_name][split_var] = operation

    logger.info(f"variables for adding subscript {subscripted_vars_dict.keys()}")
    logger.info(f"variables for merging {merg_on_variables.keys()}")
    logger.info(f"variables for splitting {split_on_variables.keys()}")

    # Will make sure all elements have been transformed
    missing_elements = (
        list(subscripted_vars_dict.keys())
        + list(merg_on_variables.keys())
        + list(split_on_variables.keys())
    )
    for section in model.sections:
        section.subscripts.append(sub_range)
        for el in section.elements:
            el: AbstractElement
            py_el_name = to_pysd_name(el.name)
            while py_el_name in missing_elements:
                missing_elements.remove(py_el_name)

            # VAriable to add the subscript
            if py_el_name in subscripted_vars_dict.keys():
                # check dependencies are subscripted or in splitter variables
                subscript_dict = subscripted_vars_dict[py_el_name]
                for c in el.components:
                    c.subscripts[0].insert(0, subscript_dim_name)
                    c.ast = add_subscript_to_abstact_syntax(
                        c.ast,
                        len(subscript_list),
                        initial=subscript_dict.get("initial", None),
                    )
            # Variable to add a merging operation
            if py_el_name in merg_on_variables.keys():
                for c in el.components:
                    c.ast = add_merge_operation_to_abstact_syntax(
                        c.ast,
                        merg_on_variables[py_el_name],
                        subscript_dim_name=subscript_dim_name,
                    )
            # Variable to add a splitting operation
            if py_el_name in split_on_variables.keys():
                for c in el.components:
                    c.ast = add_split_operation_to_abstact_syntax(
                        c.ast,
                        split_on_variables[py_el_name],
                        subscript_dim_name=subscript_dim_name,
                    )

    if missing_elements:
        raise RuntimeError(f"Elements missing were not processed {missing_elements=}")
