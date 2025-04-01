# LOTUS: Local Turbulence Closure in the Ice-Ocean Boundary Layer

LOTUS is a semi-empirical one-dimensional model of the turbulent ice-ocean boundary layer. Starting at the end, all these terms in order mean:
- **boundary layer**: Length and velocity scales of the flow are governed by its proximity to a boundary
- **ice-ocean**: exchanges of mass, heat, and momentum occur between (sea) water and (sea or freshwater) ice.
- **turbulent**: A large part of the boundary layer is turbulent, with its vertical structure dictated by eddy viscosities
- **one-dimensional**: Properties in LOTUS vary only in the vertical, with possibilities to describe the steady state or forward time stepping
- **semi-empirical**: Eddy viscosities and diffusivities follow the form $K\sim u_\ast \lambda$ with a turbulent velocity scale $u_\ast$ and a turbulent length scale $\lambda$, where $u_\star$ follows from the vertical shear in a rigorous Ekman equation and $\lambda$ is an empirical parameterization of observed mixing lengths

TODO add something about parameterized

### Parameters

# Quickstart

## Installation

### Prerequisites
- TODO check
- Python 3.8 or higher 
- pip (Python package installer)

### Steps

1. **Clone the repository:**
    ```sh
    git clone https://github.com/TODO/lotus.git
    cd TODO
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
from ltc import SteadyLTCModel
```

### Inline documentation

TODO: command to display help text on class

### Examples
<a name="examples"></a>

Examples are located in `examples/`.

Two dashboards are presented (additional dependency `panel`), e.g.:
```bash
panel serve examples/dashboard_simple.py`
```
Starts a simple dashboard:

![Dashboard](data/images/dashboard_simple.gif)

Tests, which mostly also showcase how LOTUS can be used, are located in the `ltc/` and `examples/` directories, alongside the modules they test.


# Overview

## Design principles

LOTUS is first and foremost intended to be, in descending order:
- a testbed, e.g. for sensitivity studies
- educational
- modular, easily extended or modified
- a reference implementation.

Computational efficiency has been sacrificed wherever necessary. However, LOTUS will hopefully facilitate more specialized implementations if needed e.g. for coupling to three-dimensional models.

## Infrastructure


### Parameters and introspection

LOTUS describes the model with stateful objects - parameters can be changed (almost) at will, even on existing objects, and parts of the model rerun. The benefit is flexibility but the cost is increased opacity as it becomes harder to track which algorithm affects which part of the model state. This necessitates introspection capabilities.

LOTUS makes heavy use of [param][param.holoviz.org]. Models inherit from `param.Parameterized` and describe successively complex parameters that are checked at runtime. a Its main vehicle is `IOBL` (TODO), a subclass of `ParamWatchdog`. `ParamWatchdog` can:
- log when parameters are changed, to allow introspection
- protect parameters from being changed [(see next section)](#forcedparams), to protect the user from surprises.

The `param` package introduces no other dependencies and can be removed from the codebase with little effort should the need arise. 

As  a further benefit, `param` makes creating interactive dashboards with [panel](panel.holoviz.org) a breeze (see Section [Examples](#examples)).

### Forced/dynamic parameters
<a name="forcedparams"></a>

On `ParamWatchdog` and its child classes (such as `SteadyLTCModel`), one can set multiple parameters:
- `protect_frozen` 
- `params_frozen`
- `params_dynamic`

## Acknowledgments and further reading

TODO 
The core of LOTUS is basically an implementation of Miles McPhee's 2008 book "Air-ice-sea interaction" (TODO full reference). LOTUS profited from several discussions with Miles McPhee over the years and not the least multiple MatLab scripts provided by him. LOTUS ultimately sprung out of the author's attendance and involvement in AGF-211 starting in 2010.

Work on LOTUS was funded by The Research Council of Norway project TURBOT (grant #TODO), 