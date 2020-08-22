"""
This module defines the state of a turbulent ice-ocean boundary layer. Parameters in here represent physical, observable state, subject to external forcing. Physical effects (density, mixed layer depth, shear) are only represented to the effect that they are uncontroversial and do not depend on turbulence parameters such as scaling constants or algorithms (those are defined in model.py).

The canonical variable name for instances of State or its child classes is `state`.
"""

from copy import deepcopy
import param
import gsw
import numpy as np

from ..util.time import Time

import numbers
try:
    import xarray as xr
except ImportError:
    xr = None
    warning.warn('xarray not available.')


class State(param.Parameterized):
    """
    A specific state of the environment, localized in time and space.

    Base class for the state of the boundary layer.
    """

    time = param.ClassSelector(Time, default=Time(np.nan), allow_None=False, doc="""
        Time stamp. Either a float (counting days) or a numpy datetime.""")

    latitude = param.Number(default=None, bounds=(-90., 90.), doc="""
    	Latitude (degrees N).
    	""")

    longitude = param.Number(default=None, bounds=(-180., 360.), doc="""
    	Longitude (degrees E).
    	""")

    grav = param.Number(label='g', default=9.82, doc="""
		Gravitational acceleration g (m s-2)""")

    def __init__(self, *argv, **kwarg):
        state = self
        if 'latitude' not in kwarg:
            raise ValueError('Need to supply latitude')
        if 'time' in kwarg:
            kwarg['time'] = Time(kwarg.get('time'))
        super().__init__(*argv, **kwarg)
        state.grav = gsw.grav(state.latitude, 0)


class Interface(State):
    """
    The ice-ocean interface.
    """
    # ============
    # Imposed by user
    # TBD put into forcing

    ustar0 = param.Number(default=None, doc="""
		Interface friction velocity (m s-1)""")

    z0 = param.Number(default=None, doc="""
		Ice-ocean roughness length (m)""")

    Sice = param.Number(default=None, doc="""
		Ice salinity (g kg-1)""")

    #
    Uice = param.Number(default=None, doc="""
		Ageostrophic ice velocity (m s-1).""")

    # Uice and ustar related through z0, Rossby similarity, ...

    # TBD References: Ch. 6.1
    w_perc = param.Number(default=0, doc="""
		Percolation velocity of freshwater through the ice matrix (m s-1).
        TBD: downward/upward positive??
        """)


    # TBD: split up into shortwave + rest
    # rename into something like Fh_kinematic?
    qcond = param.Number(default=0, doc="""
		Kinematic conductive heat flux through ice (K m s-1). Upward positive.""")

    # TBD imposed by user etc distinction may not be useful...
    # sort alphabetically??

    # ============
    # Imposed by user or calculated from this or previous time step)
    wb0 = param.Number(default=None, doc="""
		Interface buoyancy flux (W kg-1)""")
    # use flag to distinguish between user set and calculated?
    # can set readonly=True in init??

    hice = param.Number(default=None, doc="""
		Ice draft (m)""")

    Tff = param.Number(default=None, doc="""
		Far field temperature (degrees C)""")

    Sff = param.Number(default=None, doc="""
		Far field salinity (g kg-1)""")

    # Calculated
    #Define a wb0 that overrides the wb0 (wb0_oc) calculated in BoundaryLayer?

    wT0 = param.Number(default=None, doc="""
		Interface kinematic heat flux (K m s-1)""")

    wS0 = param.Number(default=None, doc="""
		Interface salinity flux (g kg-1 m s-1)""")

    w_interface = param.Number(default=0, doc="""
		Interface velocity, i.e. melt or freeze rate (m s-1).""")

    T0 = param.Number(default=None, doc="""
		Interface temperature (deg C)""")

    S0 = param.Number(default=None, doc="""
		Interface salinity (g kg-1)""")


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
        state = self
        L_ice = 333.55e3 # J/kg
        # Approximate interface water pressure (dbar) with ice draft (m)
        cp = gsw.cp_t_exact(
            state.S0,
            gsw.t_from_CT(state.S0, state.T0, state.hice),
            state.hice) # J/kg/K
        state.QL = L_ice/cp


class WaterColumn(State):
    """
    A turbulent 1D water column.
    """
    mld = param.Number(default=None, doc="""
		Mixed layer depth (m). calculated from temperature and salinity profiles""")

    lam_max = param.Number(default=None)

    L_obukhov = param.Number(default=None, doc="""
		Obukhov length (m), -ustar0**3/kappa/wb0""")

    ustar_pc = param.Number(default=None, doc="""
		Top-of-pycnocline friction velocity (m s-1)""")

    wb_pc = param.Number(default=None, doc="""
		Top-of-pycnocline buoyancy flux (W kg-1)""")

    mix_overturns = param.Boolean(default=True, doc="""
        If True, automatically mixes overturns after density calculation.""")

    # Gridded
    # zz: Staggered grid for mean values
    _zz_grid = ['U', 'u', 'v', 'T', 'S', 'rho']

    zz = param.Array(default=None, doc="""Distance from interface on mean value grid (m).""")

    dzz = param.Array(default=None, doc="""Cell width for mean value grid (m).""")

    # all 1-D arrays now TBD do doc string once done
    U = param.Array(default=None, doc='')
    T = param.Array(default=None, doc='')
    S = param.Array(default=None, doc='')
    rho = param.Array(default=None, doc='')

    # TBD make custom class that automatically cycles through all tracers
    # contaned in it?
    tracer = param.Array(default=None, doc="""
        Arbitrary tracer field. Must be of shape (nz, N) for grid length nz and
        N tracers.""")

    # For leap frog scheme
    T_previous = param.Array(default=None, doc='')
    S_previous = param.Array(default=None, doc='')
    U_previous = param.Array(default=None, doc='')
    tracer_previous = param.Array(default=None, doc="""

        """)

    # z: Grid for gradients, fluxes, turbulent parameters, ...
    _z_grid = [
        'Kh', 'Ks', 'Km', 'gradS', 'gradT', 'n2', 'gradu', 'gradv',
        'Ri', 'ustar', 'wb', 'lam', 'Fh',
    ]

    z = param.Array(default=None, doc="""Distance from interface on gradient value grid (m).""")

    dz = param.Array(default=None, doc="""Cell width for mean value grid (m).""")

    Kh = param.Array(default=None, doc='')
    Ks = param.Array(default=None, doc='')
    Km = param.Array(default=None, doc='')
    n2 = param.Array(default=None, doc='')
    ustar = param.Array(default=None, doc='Turbulent velocity scale (m s-1)')
    lam = param.Array(default=None, doc='Mixing length, lambda (m)')

    # weed out - don't need to drag along all these parameters.
    # can calculate a bunch of them in .finalize() or otherwise
    gradS = param.Array(default=None, doc='')
    gradT = param.Array(default=None, doc='')
    gradu = param.Array(default=None, doc='')
    gradv = param.Array(default=None, doc='')
    Ri = param.Array(default=None, doc='Richardson number (-)')
    wb = param.Array(default=None, doc='Buoyancy flux (W kg-1)')

    Fh = param.Array(default=None, doc='Heat flux (W m-2)') # TBD double-check units


    # TBD kinda unused
    # n2_lim = param.Number(default=5e-5, doc="""
    #     Critical N2 to define mixed layer depth (s-2).
    #     """)

    # mld_min = param.Number(default=4, doc="""
        # Shallowest permitted mixed layer depth (m).""")

    def __init__(self, *argv, **kwarg):
        """
        TBD
        """
        state = self
        super().__init__(*argv, **kwarg)
        if state.z[0] != 0:
            raise ValueError('Upper domain boundary must be at interface (0 m)')

    @classmethod
    def from_ctd(cls, SA, CT, zz, *argv, **kwarg):
        """
        Create an instance from vertical profile of absolute salinity
        and conservative temperature.

        zz must be given as distance from interface (m).
        TBD enable automatic interpolation using util funcs
        """
        zz = np.array(zz)
        SA = np.array(SA)
        CT = np.array(CT)

        zgrid = cls._make_zgrid(zz)

        if zgrid['z'][0] != 0:
            raise NotImplementedError('Interpolate TS accordingly')

        # TBD clean this up once done
        # Keep three columns for the leap-frog time dependent scheme
        # SA_arr = np.c_[SA, SA]
        # CT_arr = np.c_[CT, CT]

        new = cls(
            S=SA, T=CT, *argv,
            **zgrid, **kwarg,
        )

        new.update_ctd()

        return new

    @staticmethod
    def _make_zgrid(zz):
        """
        Create the flux grid z from the staggered mean value grid zz.

        Note
        ----
        TBD if z[0] does not end up at ice-ocean interface, interpolate accordingly
        """
        dzz = np.diff(zz)
        z = np.r_[
            zz[0] - 0.5*(zz[1]-zz[0]),
            zz[1:] - 0.5*dzz,
        ]
        dz = np.diff(z)
        # state.h = np.max(z) # TBD basically unused
        return dict(z=z, dz=dz, zz=zz, dzz=dzz)


    # TBD split into calc_density and mix_overturns
    def update_ctd(self, itcount=None):
        """
        Update density, stratification, mixed layer depth
        """
        state = self

        # Keep track of recursion depth
        if itcount is None:
            itcount = 0
        elif itcount > 5:
            raise RecursionError

        rho = 1000 + gsw.sigma0(state.S, state.T)
        gradrho = np.r_[np.nan, np.diff(rho)     / state.dzz]

        # Check for unstable density stratification
        sort_index = np.argsort(rho)
        in_overturn = list(np.flatnonzero(sort_index != range(len(sort_index))))

        # argsort may return re-ordering even if values are actually equal.
        # Hence also check if there actually are any unstable gradients in the
        # purported overturns

        if in_overturn and np.any(gradrho[in_overturn]<0) and self.mix_overturns:
            if any(np.isnan(rho[in_overturn])):
                raise ValueError('Found NaN density')

            msg = 'Smoothing density overturns at time: {}...'
            state.param.message(msg.format(state.time))

            def next_index_in_same_overturn(i, in_overturn):
                """
                Check if next index is still in same overturn
                This can mean at least one of two things:
                  1) The next index is immediately above on the grid
                  2) Density is still increasing
                """
                return i == in_overturn[-1]+1 or rho[i] < rho[in_overturn[-1]]

            while in_overturn:
                i = in_overturn.pop() # start at deepest point

                if next_index_in_same_overturn(i, in_overturn):
                    j = in_overturn.pop()
                    # successively go up if next index still inside overturn
                    while in_overturn and next_index_in_same_overturn(j, in_overturn):
                        j = in_overturn.pop()
                    ix = slice(j, i+1)

                    msg = '... at indices: {}'
                    state.param.message(msg.format(ix))

                    state.T[ix] = np.mean(state.T[ix])
                    state.S[ix] = np.mean(state.S[ix])

                    # TBD debug this. will probably have to insert newaxis?
                    if state.tracer is not None:
                        state.tracer[ix, ...] = np.mean(state.tracer, axis=0)[np.newaxis]

            state.update_ctd(itcount=itcount+1)

        else:
            # insert NaN at uppermost flux level z[0] == 0 m
            state.gradT = np.r_[np.nan, np.diff(state.T) / state.dzz]
            state.gradS = np.r_[np.nan, np.diff(state.S) / state.dzz]

            rhom = np.interp(state.z, state.zz, rho)
            state.n2 = state.grav * gradrho / rhom

            state.rho = rho
            state.mld = state.get_mixed_layer_depth()


    def get_mixed_layer_depth(self, mld_min=4, n2_lim=5e-5):
        """
        Arguments
        ---------
            mld_min: upper limit for mixed layer depth (m)
            n2_lim: limiting value for N^2 in the mixed layer (s-2)

        Returns
        -------
            Last depth on mean value grid in the mixed layer

        Notes
        -----
            If the entire model domain is unstratified, returns last bin.

            Returns depth of mixed layer below ice, not below ocean surface.
        """
        state = self

        zz, z, n2 = state.zz, state.z, state.n2

        with np.errstate(invalid='ignore'):
            ix_pc = (z>=mld_min) & (n2>n2_lim)

        if any(ix_pc):
            # iml is the index of the first occurrence of N2>n2_lim on the z grid
            iml = np.argmax(ix_pc) # True>False; argmax gives first occurence
            # zz grid is staggered: Get last zz value where N2<=n2_lim
            mld = zz[iml-1]
        else:
            # Entire domain is unstratified, return domain boundary (see Fig 7.1)
            mld = zz[-1]
            msg = 'Entire domain unstratified at time step: {}.'
            state.param.warning(msg.format(state.time))

        return mld


    def update_shear_driven_turbulence(self):
        """
        Update gradu, gradv, alpha, ustar
        """
        state = self

        # TBD might want to make calculation of significant Ri more rigorous
        FUZZ = 1e-12
        # Insert nan at uppermost flux level z[0] (ice-ocean interface)
        # to be consistent with grid.
        gradU = np.r_[np.nan + 1j * np.nan, np.diff(state.U) / state.dzz]

        # Gradient Richardson number Ri
        s2 = np.real(gradU)**2 + np.imag(gradU)**2
        with np.errstate(invalid='ignore'):
            # state.Ri = np.where(s2>=FUZZ, state.n2/s2, 5), 5)
            Ri = np.where(s2>=FUZZ, state.n2/s2, 5)
            state.Ri = np.where(Ri<5, Ri, 5)

        state.ustar = np.sqrt(state.Km * np.sqrt(s2))


    def get_pycnocline_forcing(self):
        """
        Extract ustar, wb, and grid index at top of pycnocline.

        Pycnocline mixing length depends on ustar and wb at top of pycnocline
        last index that is still in the mixed layer

        TBD: decide consistent top of pc criterion for both wb and ustar
        """
        state = self

        pycnocline = state.z > state.mld

        if any(pycnocline):
            ix_pc = np.argmax(pycnocline) # True>False; first occurence
            top_of_pycnocline = (state.z > state.mld) & (state.z < state.mld+5)
        else:
            ix_pc = len(state.zz)-1
            top_of_pycnocline = ix_pc

        wb_pc = np.mean(state.wb[top_of_pycnocline])
        ustar_pc = state.ustar[ix_pc]

        return ustar_pc, wb_pc, ix_pc


    def init_leapfrog(self):
        """
        Initialize the backward time step for the leap frog integrator
        """
        state = self

        state.T_previous = deepcopy(state.T)
        state.S_previous = deepcopy(state.S)
        state.U_previous = deepcopy(state.U)
        state.tracer_previous = deepcopy(state.tracer)


    def output_state(self):
        """
        Return a deep deepcopy of the current state.

        If xarray is available, convert to an xarray Dataset.
        """
        state = self

        if xr is None:
            return deepcopy(state)
        else:
            dct = dict(state.get_param_values())

            data = {}

            for k, v in dct.items():
                if v is None:
                    data[k] = (['time'], [None])
                else:
                    if isinstance(v, (numbers.Number, bool)):
                        data[k] = (['time'], [v])
                    elif k in state._z_grid:
                        data[k] = (['z', 'time'], v[:, np.newaxis])
                    elif k in state._zz_grid:
                        data[k] = (['zz', 'time'], v[:, np.newaxis])

            data['z'] = (['z'] , state.z )
            data['zz'] = (['zz'], state.zz)
            data['time'] = (['time'], [state.time.value])

            return deepcopy(xr.Dataset(data))


class IceOceanBoundaryLayer(Interface, WaterColumn):
    """
    A coupled ice-ocean boundary layer.
    """

    # TBD should go into model. does not need to be dedicated parameter...
    delU = param.Number(default=None, doc="""ustar/ velocity ratio at shallowest
        grid point (from constant stress layer equations)""")


    def calculate_ancillary(self):
        """
        TBD non-essential calculations

        can be done offline
        """
        state = self

        # TBD superfluous, write util or otherwise unify with above
        cp = gsw.cp_t_exact(
            state.S,
            gsw.t_from_CT(state.S, state.T, state.zz),
            state.zz
        )
        state.Fh = state.rho * cp * state.Kh * state.gradT

        state.Uice = state.U[0] + state.delU
