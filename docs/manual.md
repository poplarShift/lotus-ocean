# LOTUS: Local Turbulence Closure in the Ice-Ocean Boundary Layer

LOTUS is a semi-empirical one-dimensional model of the turbulent ice-ocean boundary layer (IOBL) based on the local turbulence closure (McPhee, 2008). All these terms mean:

- **local turbulence closure**: Eddy viscosities and diffusivities follow the form $K\sim u_{\ast} \lambda$ with a (local) turbulent velocity scale $u_{\ast0}$ and a turbulent length scale $\lambda$.
- **boundary layer**: The transition zone between the free flow and a rigid boundary. Length and velocity scales of the flow are governed by its proximity to this boundary.
- **ice-ocean**: Exchanges of mass, heat, and momentum occur between (sea) water and (sea or freshwater) ice.
- **turbulent**: A large part of the boundary layer is turbulent, with its vertical structure dictated by eddy viscosities.
- **one-dimensional**: Properties in LOTUS vary only in the vertical, with possibilities to describe a steady state or step the model forward in time.
- **semi-empirical**: The vertical shear and shear stress follow from a rigorous Ekman equation where $\lambda$ is an empirical parameterization of the vertical mixing length.

It should also be noted that while LOTUS focuses on the ice-ocean boundary layer, one should (with little modification) be able to use it for any oceanic boundary layer, e.g. bottom boundary layers. See `examples/annual_cycles/baffin_bay.py` for an example with open water part of the year.

# Quickstart

## Installation

### Prerequisites
LOTUS has been tested with Python 3.10. 

Core dependencies:

- gsw
- numpy
- param
- pandas

Recommended:

- xarray

For examples and tests:

- bokeh
- holoviews
- matplotlib
- panel
- pytest


### Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/poplarShift/lotus.git
    cd lotus
    ```

2. **Create a virtual environment:** (optional)

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the dependencies as desired:**
    For only the bare minimum:
    
    ```sh
    pip install -r requirements_core.txt
    ```
    For the full set
    
    ```sh
    pip install -r requirements_optional.txt
    ```
    
    Note that the versions fixed in the requirements files are the ones that LOTUS has been tested with. It may well run with other versions of the same packages too.

## Usage (quickstart)

### Running the model

```python
from lotus import SteadyLTCModel
```

### Documentation

As all models inherit from `param.Parameterized`, we can use pydoc to generate formatted summaries of individual objects.

```python
import pydoc
from lotus.iobl import SteadyLTCModel
pydoc.help(SteadyLTCModel)
```


To generate html documentation of the entire module tree, use, e.g., `pydoc`:
```bash
for mod in $(find lotus/ -name "*.py"|tr '/' '.' | rev | cut -c 4-|rev|grep -v __init__|grep -v test_)
    do pydoc -w $mod && mv $mod.html docs/
done

for mod in $(find lotus/ -type d |grep -v __pycache__ | tr '/' '.')
    do pydoc -w $mod && mv $mod.html docs/
done
```

### Examples
<a name="examples"></a>

Examples are located in `examples/`.

Two dashboards are presented (additional dependency `panel`), e.g.:
```bash
panel serve examples/dashboard_simple.py
```
starts a simple dashboard:

![A simple dashboard](data/images/dashboard_simple.png)

Tests, which mostly also showcase how LOTUS can be used, are located in the `ltc/` and `examples/` directories, alongside the modules they test.


# Overview

## Design principles

LOTUS is first and foremost intended to be, in descending order:

- a testbed, e.g. for sensitivity studies
- educational
- modular, easily extended or modified
- a reference implementation.

Computational efficiency has been sacrificed wherever necessary. However, LOTUS will hopefully facilitate more specialized implementations if needed e.g. for coupling to three-dimensional models.

That being said, LOTUS runs fast enough on modern machines to integrate a seasonal cycle at 6-hourly resolution within less than two minutes (without parameter assignment logging, see [below](#parameters)).

## Codebase structure

The basic structure of the model is as follows.

- `lotus.iobl.model.ModelState` (and its child classes) saves all variables and parameters pertaining to a turbulent oceanic boundary layer. It uses these informations to predict the vertical profile of turbulence parameters, keeping track of its output.
- `lotus.run.stepper.Stepper` (and its child classes) project `ModelState` instances into the future (using turbulent mixing coefficients and basic diffusion equations).
- `lotus.run.forcing.Forcing` contains a timeseries that is able to force instances of `ModelState` (e.g., by setting ice-ocean interface stress $u_{\ast0}$).
- `lotus.run.run.ModelRun` then repeatedly applies a stepper, taking into account variables as prescribed by `forcing`. 

Other notable modules are:

<!-- - `lotus/iobl/model.py` -->
- `lotus/iobl/rossby.py` defines the Rossby similarity scaling, an analytical model of the IOBL with neutral, stable, or unstable stratification. It comprises two layers (logarithmic in addition to an Ekman layer with constant eddy viscosity)
<!-- - `lotus/iobl/util.py`  -->
- `lotus/util/param.py` defines various classes to interact with the `param` package, such as `ParamWatchdog` that surveys when and how parameters are set, and various context managers to prevent (or allow) parameters from being changed.
- `lotus/util/profile.py` contains utilities to create CTD profiles that serve as input to `ModelState` instances.
<!-- - `lotus/util/selector.py` could just remove this from lotus? -->
- `lotus/util/time.py` defines a custom `Time`, which 

## Infrastructure


### Parameters and introspection
<a name="parameters"></a>

LOTUS describes the model with stateful objects - parameters can be changed (almost) at will, even on existing objects, and parts of the model rerun. The benefit is flexibility but the cost is increased opacity as it becomes harder to track which algorithm affects which part of the model state. This necessitates introspection capabilities.

LOTUS makes heavy use of [param](param.holoviz.org). Models inherit from `param.Parameterized` and describe successively complex parameters that are checked at runtime. Its main vehicles are `IOBL` and `SteadyLTCModel`, which inherit from `ParamWatchdog`. `ParamWatchdog` can:

- log when parameters are changed, to allow introspection
- protect parameters from being changed ([see next section](#forcedparams)), to protect the user from surprises.

The `param` package introduces no other dependencies and can be removed from the codebase with little effort should the need arise. 

As  a further benefit, `param` makes creating interactive dashboards with [panel](panel.holoviz.org) a breeze (see Section [Examples](#examples)).


The logging level can be set using e.g.
```python
import param
param.get_logger().setLevel(param.DEBUG)
```

### Forced/dynamic parameters
<a name="forcedparams"></a>

On `ParamWatchdog` and its child classes (such as `SteadyLTCModel`), one can set multiple parameters that relate to "frozen" and "dynamic" parameters:
- `params_frozen` can, once the instance is created, not easily be changed. By default, all parameters that are assigned at instance creation time will be frozen (since it is assumed they are specifically set by the user). This can be overridden using the `ParamUnfreezer` context manager, which temporarily unfreezes these parameters.
- `params_dynamic` are parameters designed to not be frozen. For instance, one may assign an ice-ocean buoyancy flux `wb0` at model creation time to initialize length scale calculations, but also allow it to be updated using `update_interface`, which solves the three-equation system in more detail.

## Acknowledgments and further reading

The core of LOTUS is essentially an implementation of Miles McPhee's 2008 book "Air-ice-sea interaction" [1]. LOTUS profited from several discussions with Miles McPhee over the years and not the least multiple MatLab scripts provided by him. LOTUS ultimately sprung out of the author's attendance and involvement in AGF-211 starting in 2010.

Work on LOTUS was funded by The Research Council of Norway project TURBOT (grant no. 352188), the Fram Centre project SUDARCO, and the Sentinel North strategy at Université Laval.

[1] M. McPhee, Air-ice-ocean Interaction: Turbulent Ocean Boundary Layer Exchange Processes. Springer New York, 2008. doi: 10.1007/978-0-387-78335-2.
