"""
This module defines the state of a turbulent ice-ocean boundary layer. Parameters in here represent physical, observable state, subject to external forcing. Physical effects (density, mixed layer depth, shear) are only represented to the effect that they are uncontroversial and do not depend on turbulence parameters such as scaling constants or algorithms (those are defined in model.py).
This module contains the core of the Local Turbulence Closure model of ice-ocean boundary layer turbulence as formulated in McPhee 2008. Parameters in here represent model parameters such as scaling constants that are not necessarily directly observable but have been constrained through through several field experiments.

"""

from copy import deepcopy
import warnings
import param
import gsw
import numpy as np

from .util import (
    get_transfer_coefficients,
    convective_vel_scale,
    alpha_nr,
    get_mixed_layer_depth_N2lim,
)
from ..util.time import Time
from ..util.param import ParamWatchdog
from ..util.profile import significant_decimals
from ltc.util.param import ParamUnfreezer

import numbers

try:
    import xarray as xr
except ImportError:
    xr = None
    warnings.warn("xarray not available.")


class ModelState(ParamWatchdog):
    """
    A specific state of the environment, localized in time and space, with adjustable parameters.

    Base class for the state of the boundary layer.

    Parent class of all models in this module. If you are looking to modify
    experiment parameters, pass them as keyword arguments into the model instance.

    Holds global parameters on the model.
    """

    time = param.ClassSelector(
        Time,
        default=Time(np.nan),
        allow_None=False,
        doc="""
        Time stamp. Either a float (counting days) or a numpy datetime.""",
    )

    latitude = param.Number(
        allow_None=False,
        bounds=(-90.0, 90.0),
        doc="""
    	Latitude (degrees N).
    	""",
    )

    longitude = param.Number(
        default=None,
        bounds=(-180.0, 360.0),
        doc="""
    	Longitude (degrees E).
    	""",
    )

    grav = param.Number(
        label="g",
        bounds=(-np.inf, np.inf),
        default=9.82,
        doc="""
		Gravitational acceleration g (m s-2)""",
    )

    adjustments = param.List(
        default=None,
        allow_None=True,
        doc="""
        List of callables that take in a `ModelState` instance and output a dictionary
        of parameters used to update itself.
        """,
    )

    def __init__(self, *argv, **kwarg):
        if "latitude" not in kwarg:
            raise ValueError("Need to supply latitude")
        if "time" in kwarg:
            kwarg["time"] = Time(kwarg.get("time"))
        # automatically add dynamic parameters like this - but wait until I've thought through API
        params_dynamic = kwarg.get("params_dynamic", [])
        params_dynamic.extend(["time"])
        kwarg["params_dynamic"] = params_dynamic

        super().__init__(*argv, **kwarg)
        self.grav = lambda: gsw.grav(self.latitude, 0)

    def adjust_model_parameters(self, **kwarg):
        """
        Adjust the model according to provided hooks.

        This routine should only be called as part of physically changing values, e.g. a
        time integration (not e.g. in iterations towards a steady state such as SLTC).
        Otherwise, it may lead to oscillations due to changing wb0 etc.
        """
        if self.adjustments is not None:
            with ParamUnfreezer(self):
                for hook in self.adjustments:
                    new_params = hook(self)
                    for key, val in new_params.items():
                        setattr(self, key, val)

            self.update_ctd()
            self.update_interface(update_wb0=True)


class IceOceanInterfaceParams(ParamWatchdog):
    """
    The ice-ocean interface.
    """

    cht = param.Number(
        default=0.0093,
        bounds=(0, np.inf),
        doc="""
        Interface heat transfer coefficient (-). For a stable or neutral boundary
        layer, this may be around 9.3e-3, for freezing ice around 5.7e-3.
        See e.g. Sec. 6.4.
        """,
    )

    chs = param.Number(
        default=0.0027,
        bounds=(0, np.inf),
        doc="""
        Interface salt transfer coefficient (-). For a stable or neutral boundary
        layer, this may be around 2.7e-4, for freezing ice around 5.7e-3.
        See e.g. Sec. 6.4.
        """,
    )

    ustar0 = param.Number(
        default=None,
        doc="""
		Interface friction velocity (m s-1)""",
    )

    z0 = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Ice-ocean roughness length (m)""",
    )

    Sice = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Ice salinity (g kg-1)""",
    )

    Uice = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Ageostrophic ice velocity (m s-1).""",
    )

    # Uice and ustar related through z0, Rossby similarity, ...

    w_perc = param.Number(
        default=0,
        bounds=(-np.inf, np.inf),
        doc="""
		Percolation velocity of freshwater through the ice matrix (m s-1).
        Positive values reflect a downward flux of freshwater. See Section 6.1.
        """,
    )

    qcond = param.Number(
        default=0,
        bounds=(-np.inf, np.inf),
        doc="""
		Kinematic conductive heat flux through ice (K m s-1). Upward positive.""",
    )

    wb0 = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
		Interface buoyancy flux (W kg-1)""",
    )

    hice = param.Number(
        default=1,
        bounds=(0, np.inf),
        doc="""
		Ice draft (m)""",
    )

    Tff = param.Number(
        default=None,
        bounds=(-273.15, np.inf),
        doc="""
		Far field conservative temperature (degrees C)""",
    )

    Sff = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Far field salinity (g kg-1)""",
    )

    wT0 = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
		Interface kinematic heat flux (K m s-1)""",
    )

    wS0 = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
		Interface salinity flux (g kg-1 m s-1)""",
    )

    w_interface = param.Number(
        default=0,
        bounds=(-np.inf, np.inf),
        doc="""
		Interface velocity, i.e. melt or freeze rate (m s-1).""",
    )

    T0 = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
		Interface temperature (deg C)""",
    )

    S0 = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Interface salinity (g kg-1)""",
    )

    use_three_eqn = param.Boolean(
        default=True,
        doc="""Whether to use the 3-eq.
        boundary conditions instead of the 2-eq. approach. The 2-eq. approach is
        equivalent to a bulk Stanton number parameterization with St* = cht.""",
    )

    QL = param.Number(
        default=83.8,
        bounds=(0, np.inf),
        doc="""Latent heat
        of melting for fresh ice divided by specific heat of seawater (K)""",
    )

    def __init__(self, *argv, **kwarg):
        super().__init__(*argv, **kwarg)
        if len(set(self.params_frozen).intersection({"ustar0", "Uice"})) != 1:
            raise ValueError("Exactly one of ustar0 or Uice must be given.")

    def _set_fresh_ice_QL(self):
        """
        Proper application of this formula would require calculating it after
        setting interface salinity and temperature (S0, T0),
        which itself depends on QL... As this formula is only a crude approximation
        involving a number of rather uncertain parameters, we do not care about such
        detail. This function is therefore never invoked in the current
        implementation. The code below is left here for the benefit of the reader.

        Reference: p. 111
        """
        raise NotImplementedError("Use get_haline_QL")
        L_ice = 333.55e3  # J/kg
        # Approximate interface water pressure (dbar) with ice draft (m)
        cp = gsw.cp_t_exact(
            self.S0, gsw.t_from_CT(self.S0, self.T0, self.hice), self.hice
        )  # J/kg/K
        self.QL = L_ice / cp

    def update_interface(self, update_wb0=False, **kwarg):
        """ "
        Calculate wT0, wS0, wb0, w_interface. wb0 only if it is in params_dynamic.

        See Figure 6.2,  7.2.
        """

        ustar0 = abs(self.ustar0)
        wb0 = self.wb0

        if wb0 >= 0:  # Neutral/stable
            uturb = ustar0
        elif wb0 < 0:  # Convecting
            # `mld` supplied later, e.g. by WaterColumn
            assert hasattr(
                self, "mld"
            ), "{} has no `mld` attribute (mixed layer depth)".format(type(self))
            uturb = convective_vel_scale(ustar0, wb0, self.cml, self.mld)
        else:
            raise ValueError("Need wb0 to be set")

        wb0, self.w_interface, self.wT0, self.wS0, self.T0, self.S0 = interface_fluxes(
            uturb=uturb,
            Tff=self.Tff,
            Sff=self.Sff,
            hice=self.hice,
            Sice=self.Sice,
            w_perc=self.w_perc,
            qcond=self.qcond,
            use_three_eqn=self.use_three_eqn,
            cht=self.cht,
            chs=self.chs,
            grav=self.grav,
            QL=self.QL,
        )
        if "wb0" in self.params_dynamic and update_wb0:
            self.param.debug("Setting wb0 as it is dynamic")
            self.wb0 = wb0


class WaterColumn(ParamWatchdog):
    """
    A turbulent 1D water column. It is defined on a staggered grid. Most easily
    instantiated using .from_ctd class method.
    """

    km_min = param.Number(
        default=1.8e-6,
        bounds=(0, np.inf),
        doc="""Minimum vertical eddy viscosity (m2 s-1).
        In practice, the molecular viscosity.""",
    )

    kh_min = param.Number(
        default=1.3e-7,
        bounds=(0, np.inf),
        doc="""Minimum vertical eddy diffusivity of heat (m2 s-1).
        In practice, molecular heat diffusivity.""",
    )

    # Find ref in book
    K_adjust = param.Number(
        default=0.005,
        bounds=(0, np.inf),
        doc="""Eddy diffusivity for N2<0 below mixed layer. This is an ad hoc parameter""",
    )

    Ks_Kh_ratio = param.Number(
        default=1,
        bounds=(-np.inf, np.inf),
        doc="""
        Ks/Kh ratio (-).""",
    )  # p. 142

    kappa = param.Number(
        default=0.4,
        bounds=(0, np.inf),
        doc="""
        von Kármán constant (-)""",
    )

    lam_star = param.Number(
        default=0.028,
        bounds=(0, np.inf),
        doc="""
        Lambda star, Λ* (-). References: p.77: 0.024""",
    )

    lam_min = param.Number(
        default=0.01,
        bounds=(0, np.inf),
        doc="""
        Minimum allowed mixing length (in meters). This is an ad hoc parameter
        to provide numerical stability in time stepping the model.""",
    )

    RC = param.Number(
        default=0.2,
        bounds=(0, np.inf),
        doc="""
        Critical flux Richardson number""",
    )

    cml = param.Number(
        default=0.2,
        bounds=(0, np.inf),
        doc="""Constant relating
        convective mixing length to mixed layer depth, around 0.2 to 0.4. See
        pp. 142""",
    )

    fcor = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
        Coriolis frequency (s-1)""",
    )

    mld = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Mixed layer depth (m) below ice-ocean interface.
        Either user-supplied or computed from state using `mld_algorithm`.""",
    )

    mld_algorithm = param.Callable(
        default=get_mixed_layer_depth_N2lim,
        allow_None=True,
        doc="""
        Callable that consumes a `state` instance and outputs mixed layer depth.""",
    )

    lam_max = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""Maximum mixing length (m), see e.g. Figure 7.2 
        """,
    )

    lam_pc = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""Top-of-pycnocline mixing length (in meters). See e.g. Figure 7.2""",
    )

    L_obukhov = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
		Obukhov length (m), -ustar0**3/kappa/wb0""",
    )

    ustar_pc = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
		Top-of-pycnocline friction velocity (m s-1)""",
    )

    wb_pc = param.Number(
        default=None,
        bounds=(-np.inf, np.inf),
        doc="""
		Top-of-pycnocline buoyancy flux (W kg-1)""",
    )

    ix_pc = param.Number(
        default=None,
        bounds=(0, np.inf),
        doc="""
        Grid index of top of pycnocline on z (flux) grid.""",
    )

    mix_overturns = param.Boolean(
        default=True,
        doc="""
        If True, automatically mixes overturns after density calculation.""",
    )

    # Gridded
    # zz: Staggered grid for mean values
    _zz_grid = ["U", "u", "v", "T", "S", "rho"]

    zz = param.Array(
        default=None, doc="""Distance from interface on mean value grid (m)."""
    )

    dzz = param.Array(default=None, doc="""Cell width for mean value grid (m).""")

    # all 1-D arrays
    U = param.Array(
        default=None,
        doc="""
        Water velocity (complex numbers; real part to the East, imaginary to the North)
        (m s-1).""",
    )
    T = param.Array(
        default=None, doc="""Water conservative temperature (TEOS-10) (deg C)"""
    )
    S = param.Array(default=None, doc="""Water absolute salinity (TEOS-10) (g kg-1)""")
    rho = param.Array(default=None, doc="""Water potential density (kg m-3)""")

    tracer = param.Array(
        default=None,
        doc="""
        Arbitrary tracer field. Must be of same shape as T, S, U.""",
    )

    # For leap frog scheme
    T_previous = param.Array(default=None, doc="Holds the previous time step's value.")
    S_previous = param.Array(default=None, doc="Holds the previous time step's value.")
    U_previous = param.Array(default=None, doc="Holds the previous time step's value.")
    tracer_previous = param.Array(
        default=None, doc="Holds the previous time step's value."
    )

    # z: Grid for gradients, fluxes, turbulent parameters, ...
    _z_grid = [
        "Kh",
        "Ks",
        "Km",
        "gradS",
        "gradT",
        "n2",
        "gradu",
        "gradv",
        "Ri",
        "ustar",
        "wb",
        "lam",
        "Fh",
    ]

    z = param.Array(
        default=None, doc="""Distance from interface on gradient value grid (m)."""
    )

    dz = param.Array(default=None, doc="""Cell width for mean value grid (m).""")

    Kh = param.Array(default=None, doc="")
    Ks = param.Array(default=None, doc="")
    Km = param.Array(default=None, doc="")
    n2 = param.Array(default=None, doc="")
    ustar = param.Array(default=None, doc="Turbulent velocity scale (m s-1)")
    lam = param.Array(default=None, doc="Mixing length, lambda (m)")

    # weed out - don't need to drag along all these parameters.
    # can calculate a bunch of them in .finalize() or otherwise
    gradS = param.Array(default=None, doc="")
    gradT = param.Array(default=None, doc="")
    gradu = param.Array(default=None, doc="")
    gradv = param.Array(default=None, doc="")
    Ri = param.Array(default=None, doc="Richardson number (-)")
    wb = param.Array(default=None, doc="Buoyancy flux (W kg-1)")

    Fh = param.Array(default=None, doc="Heat flux (W m-2)")

    def __init__(self, *argv, **kwarg):
        super().__init__(*argv, **kwarg)
        self.fcor = lambda: gsw.f(self.latitude)  # auto-update fcor
        if self.z[0] != 0:
            raise ValueError("Upper domain boundary must be at interface (0 m)")

    @classmethod
    def from_ctd(cls, SA, CT, zz, *argv, **kwarg):
        """
        Create an instance from vertical profile of absolute salinity
        and conservative temperature.

        zz must be given as distance from interface (m).
        """
        zz = np.array(zz)
        SA = np.array(SA)
        CT = np.array(CT)

        zgrid = cls._make_zgrid(zz)

        if zgrid["z"][0] != 0:
            raise NotImplementedError("Interpolate TS accordingly")

        new = cls(
            S=SA,
            T=CT,
            *argv,
            **zgrid,
            **kwarg,
        )

        new.update_ctd()

        return new

    @staticmethod
    def _make_zgrid(zz):
        """
        Create the flux grid z from the staggered mean value grid zz.

        Note
        ----
        If z[0] does not end up at ice-ocean interface, user must
        interpolate accordingly
        """
        dzz = np.diff(zz)
        z = np.r_[
            zz[0] - 0.5 * (zz[1] - zz[0]),
            zz[1:] - 0.5 * dzz,
        ]
        sig = significant_decimals(z)
        z = np.around(z, decimals=sig)

        dz = np.diff(z)
        return dict(z=z, dz=dz, zz=zz, dzz=dzz)

    def update_ctd(self, itcount=None):
        """
        Update density, stratification, mixed layer depth.

        This recurses a maximum amount of times (5) to smooth out density overturns.
        """
        # Keep track of recursion depth
        if itcount is None:
            itcount = 0
        elif itcount > 5:
            raise RecursionError

        rho = 1000 + gsw.sigma0(self.S, self.T)
        gradrho = np.r_[np.nan, np.diff(rho) / self.dzz]

        # Check for unstable density stratification
        sort_index = np.argsort(rho)
        in_overturn = list(np.flatnonzero(sort_index != range(len(sort_index))))

        # argsort may return re-ordering even if values are actually equal.
        # Hence also check if there actually are any unstable gradients in the
        # purported overturns

        if in_overturn and np.any(gradrho[in_overturn] < 0) and self.mix_overturns:
            if any(np.isnan(rho[in_overturn])):
                raise ValueError("Found NaN density")

            msg = "Smoothing density overturns {} at time: {}..."
            self.param.debug(msg.format(in_overturn, self.time))

            def next_index_in_same_overturn(i, in_overturn):
                """
                Check if next index is still in same overturn
                This can mean at least one of two things:
                  1) The next index is immediately above on the grid
                  2) Density is still increasing
                """
                return i == in_overturn[-1] + 1 or rho[i] < rho[in_overturn[-1]]

            while in_overturn:
                i = in_overturn.pop()  # start at deepest point

                if in_overturn and next_index_in_same_overturn(i, in_overturn):
                    j = in_overturn.pop()
                    # successively go up if next index still inside overturn
                    while in_overturn and next_index_in_same_overturn(j, in_overturn):
                        j = in_overturn.pop()
                    ix = slice(j, i + 1)

                    msg = "... at indices: {}"
                    self.param.debug(msg.format(ix))

                    self.T[ix] = np.mean(self.T[ix])
                    self.S[ix] = np.mean(self.S[ix])

                    if self.tracer is not None:
                        self.tracer[ix] = np.mean(self.tracer[ix])

            self.update_ctd(itcount=itcount + 1)

        else:
            # insert NaN at uppermost flux level z[0] == 0 m
            self.gradT = np.r_[np.nan, np.diff(self.T) / self.dzz]
            self.gradS = np.r_[np.nan, np.diff(self.S) / self.dzz]

            rhom = np.interp(self.z, self.zz, rho)
            self.n2 = self.grav * gradrho / rhom

            self.rho = rho
            self.set_mixed_layer_depth()

    def set_mixed_layer_depth(self):
        """
        Returns depth of mixed layer below ice (not below ocean surface).
        """
        if self.mld_algorithm is not None:
            mld = self.mld_algorithm(self)
            if mld is None:
                # Entire domain is unstratified, return domain boundary (see Fig 7.1)
                mld = self.zz[-1]
                msg = "Entire domain unstratified at time step: {}."
                self.param.log(param.INFO, msg.format(self.time))
            self.mld = mld

        elif self.mld is not None:
            self.param.message("Keeping user-supplied mixed layer depth.")
        else:
            self.param.warning("No mixed layer depth available.")

    def update_shear_driven_turbulence(self):
        """
        Update gradu, gradv, alpha, ustar
        """
        FUZZ = 1e-12  # ad-hoc parameter
        # Insert nan at uppermost flux level z[0] (ice-ocean interface)
        # to be consistent with grid.
        gradU = np.r_[np.nan + 1j * np.nan, np.diff(self.U) / self.dzz]

        # Gradient Richardson number Ri
        s2 = np.real(gradU) ** 2 + np.imag(gradU) ** 2
        with np.errstate(invalid="ignore"):
            Ri = np.where(s2 >= FUZZ, self.n2 / s2, 5)
            self.Ri = np.where(Ri < 5, Ri, 5)

        self.ustar = np.sqrt(self.Km * np.sqrt(s2))

    def set_pycnocline_forcing(self):
        """
        Set ustar, wb, and grid index at top of pycnocline.

        Pycnocline mixing length depends on ustar and wb at top of pycnocline
        (i.e., last index that is still in the mixed layer)

        NOTE: The exact location of the pycnocline can be crucial for top-of-pycnocline
        fluxes. While the author has experimented with formulations such as
        `top_of_pycnocline = (self.z > self.mld) & (self.z < self.mld + 5)`,
        and even using different criteria for ustar (smooth profile) and
        wb (often ragged), we here simply choose the original McPhee formulation.
        """
        pycnocline = self.z > self.mld

        if any(pycnocline):
            self.ix_pc = np.argmax(pycnocline)  # True>False; first occurence
        else:
            self.ix_pc = len(self.zz) - 1
            self.param.log(
                param.INFO, "No pycnocline found, returning lowest grid cell!"
            )

        self.wb_pc = self.wb[self.ix_pc]
        self.ustar_pc = self.ustar[self.ix_pc]

    def init_leapfrog(self):
        """
        Initialize the backward time step for the leap frog integrator
        """
        self.T_previous = deepcopy(self.T)
        self.S_previous = deepcopy(self.S)
        self.U_previous = deepcopy(self.U)
        self.tracer_previous = deepcopy(self.tracer)

    def output_state(self):
        """
        Return a deepcopy of the current state.

        If xarray is available, convert to an xarray Dataset.
        """

        self.calculate_ancillary()

        if xr is None:
            return deepcopy(self)
        else:
            dct = dict(self.param.get_param_values())

            data = {}

            for k, v in dct.items():
                if v is None:
                    data[k] = (["time"], [None])
                else:
                    if isinstance(v, (numbers.Number, bool)):
                        data[k] = (["time"], [v])
                    elif k in self._z_grid:
                        data[k] = (["z", "time"], v[:, np.newaxis])
                    elif k in self._zz_grid:
                        data[k] = (["zz", "time"], v[:, np.newaxis])

            data["z"] = (["z"], self.z)
            data["zz"] = (["zz"], self.zz)
            data["time"] = (["time"], [self.time.value])

            return deepcopy(xr.Dataset(data))

    def update_mixing_length(self, **kwarg):
        """
        Calculate mixing length based on ustar using empirical models such as
        `_mixing_length_stable` and `_mixing_length_unstable`.

        See Figure 7.2.
        """

        # Use eddy haline diffusivity for buoyancy flux
        self.wb = self.Ks * self.n2

        # First pass: Estimate maximum mixing length
        # using the formula for the stable case
        lam_max_similarity, self.L_obukhov = self._max_mixing_length_similarity(
            abs(self.ustar0), self.wb0
        )

        if self.wb0 >= 0:
            # Neutral or melting
            self.lam_max = lam_max_similarity
            lam = self._mixing_length_stable()
        elif self.wb0 < 0:
            """
            Adjust the mixing length to the unstable scenario.

            Upper left corner of Fig. 7.2
            """
            lam_max_unstable = self.kappa * self.mld
            if lam_max_similarity < 0 or lam_max_similarity > lam_max_unstable:
                self.lam_max = lam_max_unstable
            else:
                self.lam_max = lam_max_similarity

            lam = self._mixing_length_unstable()
        else:
            raise ValueError

        # -- Pycnocline
        self.set_pycnocline_forcing()

        self.lam_pc, _ = self._max_mixing_length_similarity(self.ustar_pc, self.wb_pc)
        lam[self.ix_pc :] = self.lam_pc

        lam[lam < self.lam_min] = self.lam_min

        self.lam = lam

    def _max_mixing_length_similarity(self, ustar, wb):
        """
        Calculate the maximum outer layer mixing length for the stable case
        using the similarity scaling.

        Also good for mixing lengths in the pycnocline,
        and serves as first guess for the unstable case.

        Returns
        -------
            Maximum mixing length
            Obukhov length
        """
        if wb == 0:
            L_obukhov = np.inf
        else:
            L_obukhov = ustar**3 / (self.kappa * wb)

        fcor = abs(self.fcor)
        if wb == 0 or ustar == 0:
            etastar2 = 1.0
        else:
            etastar2 = 1.0 / (
                1 + self.lam_star * ustar / (self.kappa * fcor * self.RC * L_obukhov)
            )

        lam_max = etastar2 * self.lam_star * ustar / fcor

        return lam_max, L_obukhov

    def _mixing_length_stable(self, **kwarg):
        """
        Calculate mixing length for neutral or stable stratification for mixed layer
        and Law of the Wall layer.

        Figure 7.2, right hand side; Figure 5.17b
        """

        lam_max = self.lam_max
        lam = self.kappa * self.z / (1 + 5 * abs(self.z) / self.L_obukhov)
        lam[lam > lam_max] = lam_max

        return lam

    def _mixing_length_unstable(self, **kwarg):
        """
        Calculate mixed-layer mixing length for a convective boundary layer.

        Figure 7.2, left hand side.
        """

        wstar = convective_vel_scale(abs(self.ustar0), self.wb0, self.cml, self.mld)

        L = wstar**3 / (self.kappa * self.wb0)

        # Start with surface layer constant stress scaling
        lam = self.kappa * abs(self.z) * (1 - abs(self.z) / L) ** 2
        lam[lam > self.lam_max] = self.lam_max

        return lam

    def update_eddy_coefficients(self, **kwarg):
        """
        Update vertical eddy viscosity and diffusivity. This is the part that gives
        the method its name, "Local Turbulence Closure":
        Km = ustar * lambda.
        """

        # now update the K fields
        Km = self.ustar * self.lam

        # updated Km is going to be used in the .solve_steady_ekman routine
        Kh = Km
        Ks = self.Ks_Kh_ratio * Kh

        # The following "Richardson number adjustment" is ad-hoc in the original scripts.
        # It is discussed on pp. 142 but little empirical evidence is available for its
        # necessity.
        alpha = alpha_nr(self.Ri)
        Ks = np.where(self.z <= self.mld, alpha * Ks, Ks)

        if self.K_adjust:
            # final ad-hoc test: if model.K_adjust is nonzero, then set all Kh values
            #  below the mixed layer for which N2 is negative to K_adjust
            with np.errstate(invalid="ignore"):
                Ks = np.where((self.z > self.mld) & (self.n2 < 0), self.K_adjust, Ks)

        # Respect minimum values
        Km[Km < self.km_min] = self.km_min
        Ks[Ks < self.kh_min] = self.kh_min

        self.Km = Km
        self.Ks = Ks
        self.Kh = self.Ks / self.Ks_Kh_ratio


class IceOceanInterface(
    ModelState,
    IceOceanInterfaceParams,
):
    """An ice-ocean interface model"""

    pass


class IOBL(
    ModelState,
    WaterColumn,
    IceOceanInterfaceParams,
):
    """
    A turbulent ice-ocean boundary layer coupling the water column with the ice-ocean interface.
    """

    # Something like the following could easily be implemented in _solve_steady_ekman.
    # U_source = param.ClassSelector(
    #     class_=(np.ndarray, complex, float),
    #     default=None,
    #     doc="""
    #     Source function for the horizontal velocity (m s-2). May accomodate a
    #     thermal wind. Treats source as barotropic
    #     if given a (complex or real) number.
    #     """,
    # )

    T_btm_bc = param.Number(
        default=None,
        doc="""
        Temperature bottom boundary condition. Can be either a fixed value (deg C)
        or a flux (K m s-1, upward positive). Select using T_btm_bc_kind.)
        """,
    )

    T_btm_bc_kind = param.ObjectSelector(
        default="fixed",
        objects=["flux", "value", "fixed"],
        doc="""
            Select the meaning of the T_btm_bc parameter. 'fixed' keeps the
            previous' time step's bottom temperature.
            """,
    )

    S_btm_bc = param.Number(
        default=None,
        doc="""
        Salinity bottom boundary condition. Can be either a fixed value (g kg-1)
        or a flux (g kg-1 m s-1, upward positive). Select using S_btm_bc_kind.)
        """,
    )

    S_btm_bc_kind = param.ObjectSelector(
        default="fixed",
        objects=["flux", "value", "fixed"],
        doc="""
            Select the meaning of the S_btm_bc parameter. 'fixed' keeps the
            previous' time step's bottom salinity.
            """,
    )

    tracer_btm_bc = param.Number(
        default=None,
        doc="""
        Tracer bottom boundary condition. Can be either a fixed value
        or a flux (upward positive). Select using tracer_btm_bc_kind.)
        """,
    )

    tracer_btm_bc_kind = param.ObjectSelector(
        default="fixed",
        objects=["flux", "value", "fixed"],
        doc="""
            Select the meaning of the tracer_btm_bc parameter. 'fixed' keeps the
            previous' time step's bottom value.
            """,
    )

    T_source = param.Array(default=None, doc="")
    S_source = param.Array(default=None, doc="")
    tracer_source = param.Array(default=None, doc="")

    def __init__(self, *argv, **kwarg):
        super().__init__(*argv, **kwarg)
        self.Sff = lambda: self.S[0]  # auto-update
        self.Tff = lambda: self.T[0]  # auto-update

    def finalize_interface_shear(self, **kwarg):
        """
        Shear at interface is governed by Law of the Wall.

        Depending on whether ustar0 or Uice is prescribed, the other is calculated.

        OBS: This must only be called after a suitable current profile has been found
        (see e.g. steady_ltc).
        """

        logratio = np.log(abs(self.zz[0]) / self.z0) / self.kappa
        if "ustar0" in self.params_frozen:
            delU = self.ustar0 * logratio
            self.Uice = self.U[0] + delU
        elif "Uice" in self.params_frozen:
            # This makes the ustar0-Uice correspondence exact - but note they have already
            # been tuned to each other in find_steady_k, but with a looser
            # convergence criterion (tolerance 5e-3).
            delU = self.Uice - self.U[0]
            self.ustar0 = delU / logratio

    def update_turbulence(self, **kwarg):

        self.update_shear_driven_turbulence()  # new ustar
        self.update_mixing_length()  # new lambda
        self.update_eddy_coefficients()  # new Km, Kh, Ks

    def init_ekman_spiral(self, **kwarg):
        """
        Make a reasonable guess for a boundary layer with a constant
        eddy viscosity. Does not include a pycnocline or a Law-of-the-Wall regime.
        """
        if self.ustar0 is None and "Uice" in self.params_frozen:
            # use some reasonable sea ice drag coefficient to get first estimate
            self.ustar0 = self.Uice * np.sqrt(4e-3)

        if self.wb0 >= 0:
            lam_max, _ = self._max_mixing_length_similarity(abs(self.ustar0), self.wb0)
        elif self.wb0 < 0:
            lam_max = self.cml * self.mld

        # First guess at K: in neutral layer u* will decay exponentially

        self.lam = lam_max * np.ones(np.shape(self.z))
        Kmax = abs(self.ustar0) * lam_max

        K = Kmax * np.ones(np.shape(self.z))
        self.Km = deepcopy(K)
        self.Kh = deepcopy(K)
        self.Ks = deepcopy(K)
        self.lam_max = lam_max

        # initialize free Ekman spiral U = U0 * exp(-delta*z)
        delta = np.sqrt(1j * self.fcor / Kmax)
        tau0 = abs(self.ustar0) * self.ustar0
        U0 = delta * tau0 / (1j * self.fcor)

        self.U = U0 * np.exp(-delta * self.zz)
        # state.solve_steady_ekman() # u, v, U
        self.update_shear_driven_turbulence()

    def calculate_ancillary(self):
        """
        Non-essential calculations. Should be done offline (not in time loop).
        """
        cp = gsw.cp_t_exact(self.S, gsw.t_from_CT(self.S, self.T, self.zz), self.zz)
        self.Fh = self.rho * cp * self.Kh * self.gradT


class SteadyLTCModel(IOBL):
    """
    Iteratively solves the steady state Ekman equation using
    the Local Turbulence Closure model.

    Can be subclassed to define other boundary layer models.
    The key functions to override are:

    update_interface, update_mixing_length, update_eddy_coefficients
    """

    maxit = param.Integer(
        default=20,
        doc="""
        Maximum number of iterations of the steady LTC""",
    )

    tol = param.Number(
        default=0.01,
        doc="""
        Convergence tolerance for the steady LTC.""",
    )

    def find_steady_k(self, **kwarg):
        """
        Find an iterative solution to the boundary layer problem.
        """

        for j in range(self.maxit):
            Km_old = deepcopy(self.Km)
            self.update_turbulence()
            self._solve_steady_ekman()  # calculate new currents
            self.update_interface()
            # Do at least two iterations
            if j > 2 and self._has_converged(self.Km, Km_old):
                self.param.debug(
                    "Model at time {} converged after {} iterations.".format(
                        self.time, j
                    )
                )
                if "ustar0" in self.params_frozen:
                    # stress already matches the frozen value by definition
                    self.param.debug("ustar0 already prescribed, no need to adjust.")
                    return
                elif "Uice" in self.params_frozen:
                    self.param.debug("Uice prescribed, adjusting ustar0.")
                    # still need to hit target ice-ocean relative velocity
                    logratio = np.log(abs(self.zz[0]) / self.z0) / self.kappa
                    delU = self.ustar0 * logratio
                    Uice_reached = (
                        self.U[0] + delU
                    )  # the Uice we would get with current value of ustar0
                    ratio = Uice_reached / self.Uice
                    if abs(ratio - 1) > 5e-3:
                        self.ustar0 /= ratio
                    else:
                        # stress matches the target value
                        self.param.debug(
                            "ustar0 converged after {} iterations.".format(j)
                        )
                        return

        self.param.warning("Model at time {} did not converge!".format(self.time))

    def _has_converged(self, Km, Km_old):
        if np.nanmean(abs((Km - Km_old) / Km)) < self.tol:
            return True
        else:
            return False

    def _solve_steady_ekman(self, **kwarg):
        """
        Solve the steady Ekman equation for horizontal velocity.

        Solves 0 = (Km * u_z)_z - ifu for a given Km profile.

        Ch. 7.3
        """

        nz = len(self.z)

        # init
        A = np.zeros(nz - 1, dtype=complex)
        B = np.zeros(nz - 1, dtype=complex)
        C = np.zeros(nz - 1, dtype=complex)
        D = np.zeros(nz - 1, dtype=complex)
        E = np.zeros(nz - 1, dtype=complex)

        # The equations in chapter 7 are formulated for z upward positive.
        # To avoid confusion we adopt that convention in this integration:
        dz, dzz = -self.dz, -self.dzz

        tau0 = abs(self.ustar0) * self.ustar0

        A[0] = self.Km[1] / (dz[0] * dzz[0])
        ix = slice(1, nz - 1)
        ix_plus_1 = slice(2, nz)
        ix_minus_1 = slice(0, nz - 2)
        A[ix] = self.Km[ix_plus_1] / (dz[ix] * dzz[ix])
        C[ix] = self.Km[ix] / (dz[ix] * dzz[ix_minus_1])
        B[ix] = -(1j * self.fcor + A[ix] + C[ix])

        D[0] = -tau0 / (dz[0] * (A[0] + 1j * self.fcor))
        E[0] = A[0] / (A[0] + 1j * self.fcor)
        for i in range(1, nz - 1):
            denominator = B[i] + C[i] * E[i - 1]
            D[i] = -C[i] * D[i - 1] / denominator
            E[i] = -A[i] / denominator

        # Eqs. 7.25, 7.6
        # Fix ageostrophic background to 0
        u = np.zeros(nz, dtype=complex)
        for i in range(nz - 2, -1, -1):
            u[i] = D[i] + E[i] * u[i + 1]

        self.U = u

    def steady_ltc(self):
        self.init_ekman_spiral()
        self.find_steady_k()
        self.update_ctd()
        self.finalize_interface_shear()


def get_haline_QL(QL, Sice):
    """
    Maykut (1985, his equations 6.13 and 19)
    """

    return QL * (1 - 0.03 * Sice)


def interface_fluxes(
    # nomenclature as in the parameter definitions above
    uturb,
    Tff,
    Sff,
    hice,
    Sice,
    w_perc,
    qcond,
    use_three_eqn,
    cht,
    chs,
    grav,
    QL,
):
    """ "
    Calculate wT0, wS0, wb0, w_interface using the 2- or 3-equation formulas.
    This function can handle vectors.

    See Figures 6.2,  7.2
    """
    QL_haline = get_haline_QL(QL=QL, Sice=Sice)
    if use_three_eqn:
        # Eqns 6.8, 6.9
        # Neglect difference between interface pressure and ice draft
        m = -gsw.t_freezing(Sff, hice, 1) / Sff
        T_L = QL_haline * chs / cht
        gamma_2 = w_perc / cht / uturb
        AA = m * (1 + gamma_2)
        T_H = Tff - qcond / cht / uturb - m * gamma_2 * Sice

        BB = T_H + T_L + gamma_2 * QL_haline - AA * Sice
        CC = -(Sice * T_H + T_L * Sff + gamma_2 * QL_haline * Sice)
        # solve AA*S0^2+BB*S0+CC=0
        S0 = (-BB + np.sqrt(BB**2 - 4 * AA * CC)) / 2 / AA
        T0 = -m * S0
        wT0 = cht * uturb * (Tff - T0)
        wS0 = chs * uturb * (Sff - S0)
        w_interface = wS0 / (S0 - Sice) - w_perc

    else:
        S0 = Sff
        # Neglect difference between interface pressure and ice draft
        T0 = gsw.t_freezing(S0, hice, 1)
        wT0 = cht * uturb * (Tff - T0)
        w_interface = (wT0 - qcond) / QL_haline
        wS0 = (w_interface + w_perc) * (S0 - Sice)

    # Neglect difference between pressure and depth of ice draft
    alpha = gsw.alpha(S0, T0, hice)
    beta = gsw.beta(S0, T0, hice)
    wb0 = grav * (beta * wS0 - alpha * wT0)

    T0 = np.where(uturb == 0, Tff, T0)
    S0 = np.where(uturb == 0, Sff, S0)
    wT0 = np.where(uturb == 0, 0, wT0)
    wS0 = np.where(uturb == 0, 0, wS0)
    wb0 = np.where(uturb == 0, 0, wb0)
    w_interface = np.where(uturb == 0, 0, w_interface)

    return wb0, w_interface, wT0, wS0, T0, S0
