"""
This module contains the core of the Local Turbulence Closure model of ice-ocean boundary layer turbulence as formulated in McPhee 2008. Parameters in here represent model parameters such as scaling constants that are not necessarily directly observable but have been constrained through through several field experiments.

The canonical variable name for instances of SteadyLTC is `model`.
"""

# TBD need to address changes in `hice`:

# TBD put all of these in one single class? or otherwise split up by functional units
#

# TBD weed through methods and make everything nonessential private!!

from copy import deepcopy
import param
import gsw
import numpy as np

from ..iobl.state import State
from .util import get_transfer_coefficients, convective_vel_scale, alpha_nr
from ..util.time import Time


class Model(param.Parameterized):
    """
    Parent class of all models in this module. If you are looking to modify
    experiment parameters, pass them as keyword arguments into the model instance.

    Holds global parameters on the model.
    TBD move parameters to where they are needed!
    """

    state = param.ClassSelector(State, allow_None=False, doc="""""")

    # Refs
    cht = param.Number(default=None, doc="""
		Interface heat transfer coefficient (-). For a stable or neutral boundary
        layer, this may be around 9.3e-3, for freezing ice around 5.7e-3.
        """)

    chs = param.Number(default=None, doc="""
		Interface salt transfer coefficient (-). For a stable or neutral boundary
        layer, this may be around 2.7e-4, for freezing ice around 5.7e-3.
        """)

    latitude = param.Number(default=None, bounds=(-90., 90.), doc="""
    	Latitude (degrees N).
    	""")

    fcor = param.Number(default=None, doc="""
		Coriolis frequency (s-1)""")

    # TBD
    km_min = param.Number(default=1.8e-6, doc="""
		Minimum vertical eddy viscosity (m2 s-1). In practice, the molecular viscosity.""")

    kh_min = param.Number(default=1.3e-7, doc="""
		Minimum vertical eddy diffusivity of heat (m2 s-1). In practice, molecular heat diffusivity.""") #TBD double check!

    # TBD are these even used??
    km_max = param.Number(default=2, doc="""
		km_max""")

    kh_max = param.Number(default=2, doc="""
		kh_max""")

    # Find ref in book
    K_adjust = param.Number(default=0.005, doc="""
		Eddy diffusivity for N2<0 below mixed layer""")

    Ks_Kh_ratio = param.Number(default=1, doc="""
		Ks/Kh ratio (-).""") # p. 142

    kappa = param.Number(default=0.4, doc="""
		von Kármán constant (-)""")

    lam_star = param.Number(default=0.028, doc="""
		Lambda star, Λ* (-). References: p.77: 0.024""")

    RC = param.Number(default=0.2, doc="""
		Critical flux Richardson number""")

    cml = param.Number(default=0.2, doc="""
		Constant relating convective mixing length to  mixed layer depth, around 0.2 to 0.4""") # pp. 142

    use_three_eqn = param.Boolean(default=True, doc="""
        Whether to use the 3-eq. boundary conditions instead of the 2-eq. approach.
        The 2-eq. approach is equivalent to a bulk Stanton number parameterization with St* = cht.""")

    QL = param.Number(default=83.8, doc="""
        Latent heat of melting for fresh ice divided by specific heat of seawater (K)""")

    adjustments = param.List(
        default=None, allow_None=True, doc="""
        List of callables that take in a `state` instance and output a dictionary of
        parameters used to update the model.
        """)

    def __init__(self, *argv, **kwarg):
        model = self
        super().__init__(*argv, **kwarg)
        model._update_latitude()

    # TBD how to implement hooks in param?
    def _update_latitude(self):
        model = self
        model.fcor = gsw.f(model.latitude)

    # TBD rename: adjust_model_parameters_parameters
    def adjust_model_parameters(self, **kwarg):
        """
        Tune the model according to the state.
        """
        model = self
        state = self.state

        if model.adjustments is not None:
            for hook in model.adjustments:
                new_params = hook(state)
                for key, val in new_params.items():
                    setattr(model, key, val)

            model.update_ctd()
            model.update_interface(**kwarg)


    def update_ctd(self):
        self.state.update_ctd()


class IceOceanInterfaceModel(Model):
    """
    Calculate ice-ocean fluxes.
    """

    def update_interface(self, **kwarg):
        """"
        Calculate wT0, wS0, wb0, w_interface

        See Figure 6.2,  7.2
        """
        model = self
        state = self.state

        model._update_farfield_TS(state)

        cht, chs = model.cht, model.chs
        ustar0 = abs(state.ustar0)
        wb0 = state.wb0

        # # TBD find better solution to do this check...
        # if wb0 is None or np.isnan(wb0):
        #     state.param.warning('Setting wb0 to 0')
        #     wb0 = 0
        # TBD
        # also check if we need iterative solution re-calculate wb0_oc
        # with re-calculated wb0...

        if wb0 >= 0: # Neutral/stable
            uturb = ustar0
        elif wb0 < 0: # Convecting
            # TBD cannot be called on Interface itself (need for mld)
            uturb = convective_vel_scale(ustar0, wb0, model.cml, state.mld)
        else:
            raise ValueError('Need wb0 to be set')

        # uturb=wstar

        if uturb == 0:
            # set T0, S0 to far field, all fluxes to zero
            state.T0 = state.Tff
            state.S0 = state.Sff
            state.wT0 = 0
            state.wS0 = 0
            state.wb0 = 0
        else:
            # TBD use bulk stanton number? provide util that
            # gives wT0, wS0, wb0 from St* parameterization?
            QL = model.get_haline_QL(state=state)
            if model.use_three_eqn:
                # Eqns 6.8, 6.9
                # Neglect difference between interface pressure and ice draft
                m = -gsw.t_freezing(state.Sff, state.hice, 1)/state.Sff
                T_L = QL*chs/cht
                gamma_2 = state.w_perc/cht/uturb
                AA = m*(1+gamma_2)
                T_H = state.Tff-state.qcond/cht/uturb-m*gamma_2*state.Sice

                BB = T_H+T_L+gamma_2*QL-AA*state.Sice
                CC = -(state.Sice*T_H+T_L*state.Sff+gamma_2*QL*state.Sice)
                # solve AA*S0^2+BB*S0+CC=0
                state.S0 = (-BB+np.sqrt(BB**2-4*AA*CC))/2/AA
                state.T0 = -m*state.S0
                state.wT0 = cht*uturb*(state.Tff-state.T0)
                state.wS0 = chs*uturb*(state.Sff-state.S0)
                state.w_interface = state.wS0/(state.S0-state.Sice)-state.w_perc

            else:
                state.S0 = state.Sff
                # Neglect difference between interface pressure and ice draft
                state.T0 = gsw.t_freezing(state.S0, state.hice, 1)
                state.wT0 = cht*uturb*(state.Tff-state.T0)
                state.w_interface = (state.wT0-state.qcond)/QL
                state.wS0 = (state.w_interface+state.w_perc) * (state.S0-state.Sice)

            # TBD need to distinguish this one from the one given at instantiation
            # if given!
            # Could do this by keeping two wb0:
            # forcing.wb0 (has precedence for mixing length calc),
            # state.wb0 (calculated)

            # Neglect difference between pressure and depth of ice draft
            alpha = gsw.alpha(state.S0, state.T0, state.hice)
            beta = gsw.beta(state.S0, state.T0, state.hice)
            state.wb0 = state.grav*(beta * state.wS0 - alpha * state.wT0)


    def get_haline_QL(self, **kwarg):
        """
        TBD refactor this to model? In that case take Sice from forcing, not
        self or model

        Maykut (1985, his equations 6.13 and 19)
        """
        model = self
        state = self.state

        return model.QL * (1 - 0.03*state.Sice)


    def _update_farfield_TS(self, state):
        """
        Update interface far-field T, S with uppermost grid cell of current state.
        """
        state.Sff = state.S[0]
        state.Tff = state.T[0]


    def _update_interface_shear(self, **kwarg):
        """
        TBD calculate with aux?

        TBD Mutually determine ustar0/Uice as given by user/forcing

        Shear at interface is governed by Law of the Wall.
        """
        state = self.state

        logratio = np.log(abs(state.zz[0]) / state.z0) / model.kappa
        state.delU = state.ustar0 * logratio


class MixingLengthModel(Model):
    """
    Calculate a vertical profile of mixing lengths.
    """

    def update_mixing_length(self, **kwarg):
        """
        Calculate mixing length based on ustar using empirical models such as
        `_mixing_length_stable` and `_mixing_length_unstable`.

        See Figure 7.2.
        """
        model = self
        state = self.state

        # Use eddy haline diffusivity for buoyancy flux
        state.wb = state.Ks * state.n2

        # First pass: Estimate maximum mixing length
        # using the formula for the stable case
        lam_max_similarity, state.L_obukhov = model._max_mixing_length_similarity(abs(state.ustar0), state.wb0)

        if state.wb0 >= 0:
            # Neutral or melting
            state.lam_max = lam_max_similarity
            lam = model._mixing_length_stable(**kwarg)
        elif state.wb0 < 0:
            """
            Adjust the mixing length to the unstable scenario.

            Upper left corner of Fig. 7.2
            """
            lam_max_unstable = model.kappa * state.mld
            if lam_max_similarity < 0 or lam_max_similarity > lam_max_unstable:
                state.lam_max = lam_max_unstable
            else:
                # TBD double check this, why whould you not set lam_max to kappa*mld throughout??
                state.lam_max = lam_max_similarity

            lam = model._mixing_length_unstable(**kwarg)
        else:
            raise ValueError

        # -- Pycnocline
        ustar_pc, wb_pc, ix_pc = state.get_pycnocline_forcing()

        lam_max_pc, _ = model._max_mixing_length_similarity(ustar_pc, wb_pc)
        lam[ix_pc:] = lam_max_pc

        # TBD make lam_min a parameter?
        lam[lam<0.01] = 0.01

        state.lam      = lam
        state.ustar_pc = ustar_pc
        state.wb_pc    = wb_pc


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
        model = self

        if wb == 0:
            lobuk = np.inf
        else:
            lobuk = ustar**3 / (model.kappa * wb)

        fcor = abs(model.fcor)
        if wb == 0:
            etastar2 = 1.
        else:
            etastar2 = 1. / (
                1 + model.lam_star * ustar /
                (model.kappa * fcor * model.RC * lobuk)
            )

        lam_max = etastar2 * model.lam_star * ustar / fcor

        return lam_max, lobuk


    def _mixing_length_stable(self, **kwarg):
        """
        Calculate mixed-layer mixing length for neutral or stable stratification.

        Figure 7.2, right hand side.
        """
        model = self
        state = self.state

        # TBD rename such that name reflects that this is for mixed layer incl.
        # surface layer / Law of the Wall
        lam_max = state.lam_max
        lam = model.kappa * state.z #TBD Fig. 5.17b
        # / (1 + 5*abs(state.z)/state.L_obukhov)
        lam[lam>lam_max] = lam_max

        return lam


    def _mixing_length_unstable(self, **kwarg):
        """
        Calculate mixed-layer mixing length for a convective boundary layer.

        Figure 7.2, left hand side.
        """
        model = self
        state = self.state

        wstar = convective_vel_scale(abs(state.ustar0), state.wb0, model.cml, state.mld)

        # TBD Sort of enhanced Obukhov scale... I guess
        # find book ref
        L = wstar**3 / (model.kappa * state.wb0)

        # Start with surface layer constant stress scaling
        lam = model.kappa * abs(state.z) * (1-abs(state.z)/L)**2
        lam[lam>state.lam_max] = state.lam_max

        return lam


class EddyCoefficientModel(Model):
    """
    Combines mixing length and ustar profile to eddy viscosity and
    eddy diffusivities.
    """

    def update_eddy_coefficients(self, **kwarg):
        """
        Update vertical eddy viscosity and diffusivity.

        LTC refers mostly to the part where Km = ustar * lambda.

        TBD add internal wave spectra in here? or when ustar is computed? How to do this in external module?
        """
        model = self
        state = self.state

        # --- ACTUAL LTC
        # now update the K fields
        Km = state.ustar * state.lam

        # TBD
        # insert background / IW-driven mixing??

        # TBD zero bottom boundary flux of T, S seems to force zero gradient and
        # bottom, hence zero n2...

        # epsIW = 1e-8
        # KmIW = 0.2*epsIW/np.nanmean(state.n2[state.z>state.mld])
        # # add background below mixed layer (where Gamma=0.2 is appropriate)
        # Km = np.where(state.z>state.mld, KmIW+Km, Km)

        # TBD clean up mess
        #  updated Km is going to be used in the .solve_steady_ekman routine
        Kh = Km
        Ks = model.Ks_Kh_ratio * Kh

        # TBD --- do some ad hoc (?) adjustments

        # TBD find justification in book, and implement
        # TBD something seems to be missing here
        # if buoyancy flux is negative, but u*0^3 > lam_max*abs(<wb>_0) then apply
        #   the Richardson number adjustment to the eddy diffusivity

        # if ri_adjust: # TBD was set to True in miles' scripts
        # TBD alpha is weird...
        alpha = alpha_nr(state.Ri)
        Ks = np.where(state.z<=state.mld, alpha*Ks, Ks)

        with np.errstate(invalid='ignore'):
            # final test: if model.K_adjust is nonzero, then set all Kh values
            #  below the mixed layer for which N2 is negative to K_adjust
            if model.K_adjust:
                Ks = np.where((state.z>state.mld) & (state.n2<0), model.K_adjust, Ks)

        # Respect minimum values
        Km[Km<model.km_min] = model.km_min
        Ks[Ks<model.kh_min] = model.kh_min

        state.Km = Km
        state.Ks = Ks
        state.Kh = state.Ks/model.Ks_Kh_ratio


class SteadyLTC(
    IceOceanInterfaceModel, # provides update_interface
    MixingLengthModel, # provides update_mixing_length
    EddyCoefficientModel, # provides update_eddy_coefficients
    Model, # basic parameters
):
    """
    Iteratively solves the steady state Ekman equation using
    the Local Turbulence Closure model.

    Can be subclassed to define other boundary layer models.
    The key functions to override are:

    update_interface, update_mixing_length, update_eddy_coefficients

    TBD
    Clearly define in a central place: Which attributes those three update_*
    methods actually update

    """

    maxit = param.Integer(default=20, doc="""
		Maximum number of iterations of the steady LTC""")

    tol = param.Number(default=0.01, doc="""
		Convergence tolerance for the steady LTC.""")

    # TBD implement into _solve_steady_ekman !!
    U_source = param.ClassSelector(class_=(np.ndarray, complex, float),
        default=None, doc="""
        Source function for the horizontal velocity (m s-2). May accomodate a
        thermal wind. Treats source as barotropic
        if given a (complex or real) number.
        """)

    # TBD make own Source/BoundaryCondition class??
    T_btm_bc = param.Number(default=None, doc="""
        Temperature bottom boundary condition. Can be either a fixed value (deg C)
        or a flux (K m s-1, upward positive). Select using T_btm_bc_kind.)
        """)

    T_btm_bc_kind = param.ObjectSelector(
        default='fixed', objects=['flux', 'value', 'fixed'], doc="""
            Select the meaning of the T_btm_bc parameter. 'fixed' keeps the
            previous' time step's bottom temperature.
            """)

    S_btm_bc = param.Number(default=None, doc="""
        Salinity bottom boundary condition. Can be either a fixed value (g kg-1)
        or a flux (g kg-1 m s-1, upward positive). Select using S_btm_bc_kind.)
        """)

    S_btm_bc_kind = param.ObjectSelector(
        default='fixed', objects=['flux', 'value', 'fixed'], doc="""
            Select the meaning of the S_btm_bc parameter. 'fixed' keeps the
            previous' time step's bottom salinity.
            """)

    tracer_btm_bc = param.Number(default=None, doc="""
        Tracer bottom boundary condition. Can be either a fixed value
        or a flux (upward positive). Select using tracer_btm_bc_kind.)
        """)

    tracer_btm_bc_kind = param.ObjectSelector(
        default='fixed', objects=['flux', 'value', 'fixed'], doc="""
            Select the meaning of the tracer_btm_bc parameter. 'fixed' keeps the
            previous' time step's bottom value.
            """)

    # TBD put into state...??
    T_source = param.Array(default=None, doc='')
    S_source = param.Array(default=None, doc='')
    tracer_source = param.Array(default=None, doc='')


    def update_turbulence(self, **kwarg):
        model = self
        state = self.state

        state.update_shear_driven_turbulence() # new ustar
        model.update_mixing_length(**kwarg) # new lambda
        model.update_eddy_coefficients(**kwarg) # new Km, Kh, Ks


    def find_steady_k(self, **kwarg):
        """
        Find an iterative solution to the boundary layer problem.
        """
        model = self
        state = self.state

        for j in range(model.maxit):
            old_state = deepcopy(state)
            model.update_turbulence(**kwarg)
            model._solve_steady_ekman(**kwarg) # calculate new currents
            # Do at least two iterations
            if j > 2 and model._has_converged(state, old_state):
                return

        model.param.warning('Model at time {} did not converge!'.format(
            state.time
        ))


    def _has_converged(self, state, old_state):
        model = self
        if np.nanmean(abs((state.Km - old_state.Km) / state.Km)) < model.tol:
            return True
        else:
            return False


    def _solve_steady_ekman(self, **kwarg):
        """
        Solve the steady Ekman equation for horizontal velocity.

        Solves 0 = (Km * u_z)_z - ifu for a given Km profile.

        Ch. 7.3
        """
        model = self
        state = self.state

        if model.U_source is not None:
            raise NotImplementedError('Does not currently respect U_source') #TBD

        nz = len(state.z)

        # init
        A = np.zeros(nz-1, dtype=complex)
        B = np.zeros(nz-1, dtype=complex)
        C = np.zeros(nz-1, dtype=complex)
        D = np.zeros(nz-1, dtype=complex)
        E = np.zeros(nz-1, dtype=complex)

        # The equations in chapter 7 are formulated for z upward positive.
        # To avoid confusion we adopt that convention in this integration:
        dz, dzz = -state.dz, -state.dzz

        tau0 = abs(state.ustar0) * state.ustar0

        A[0] = state.Km[1]/(dz[0]*dzz[0])
        ix = slice(1, nz-1)
        ix_plus_1 = slice(2, nz)
        ix_minus_1 = slice(0, nz-2)
        A[ix] = state.Km[ix_plus_1] / (dz[ix] * dzz[ix])
        C[ix] = state.Km[ix] / (dz[ix] * dzz[ix_minus_1])
        B[ix] = -(1j * model.fcor + A[ix] + C[ix])

        D[0] = -tau0/(dz[0] * (A[0] + 1j * model.fcor))
        E[0] = A[0]/(A[0] + 1j*model.fcor)
        for i in range(1, nz-1):
            denominator = B[i]+C[i]*E[i-1]
            D[i] = -C[i]*D[i-1] / denominator
            E[i] = -A[i] / denominator

        # Eqs. 7.25, 7.6
        # Fix ageostrophic background to 0
        u = np.zeros(nz, dtype=complex)
        for i in range(nz-2, -1, -1):
            u[i]=D[i] + E[i] * u[i+1]

        state.U = u


    def init_ekman_spiral(self, **kwarg):
        """
        Make a reasonable guess for a boundary layer with a constant
        eddy viscosity. Does not include a pycnocline or a Law-of-the-Wall regime.
        """
        model = self
        state = self.state

        if state.wb0 >= 0:
            lam_max, _ = model._max_mixing_length_similarity(abs(state.ustar0), state.wb0)
        elif state.wb0 < 0:
            lam_max = model.cml * state.mld

        # First guess at K: in neutral layer u* will decay exponentially

        state.lam = lam_max * np.ones(np.shape(state.z))
        Kmax = abs(state.ustar0) * lam_max

        K = Kmax * np.ones(np.shape(state.z))
        state.Km = deepcopy(K)
        state.Kh = deepcopy(K)
        state.Ks = deepcopy(K)
        state.lam_max = lam_max

        # initialize free Ekman spiral U = U0 * exp(-delta*z)
        delta = np.sqrt(1j*model.fcor / Kmax)
        tau0 = abs(state.ustar0) * state.ustar0
        U0 = delta * tau0 / (1j * model.fcor)

        state.U = U0 * np.exp(-delta * state.zz)
        # state.solve_steady_ekman() # u, v, U
        state.update_shear_driven_turbulence()
