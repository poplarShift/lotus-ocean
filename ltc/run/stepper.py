"""
This module defines how to step the boundary layer model forward in time. It
implements the leap frog integration scheme for vertical profiles of velocity,
salinity, temperature, and arbitrary tracers.
"""

from copy import deepcopy
import param
import numpy as np

from .forcing import Forcing
from ..iobl.state import State
from ..iobl.model import Model

class Stepper(param.Parameterized):
    """
    Base class for models that evolve scalar and/or momentum fields
    using a leap frog scheme.
    Not useful to instantiate; use one of its subclasses instead.

    Subclasses will have to implement the `_step_fields` and `_update_state` methods.

    TBD:
    Allows specifying viscosities for arbitrary scalar tracer fields.

    Notes
    -----
        Can presumably be sped up by vectorizing `step_field`.
    """

    model = param.ClassSelector(Model, allow_None=False, doc="""
        A model that calculates turbulence parameters for a given state.""")

    forcing = param.ClassSelector(Forcing, default=None, doc="""""")

    deltat = param.Number(default=6, doc="""
		Time step (h)""")

    # see e.g. shampine2009stability
    # - there are various ways of dealing with the numerical instability
    # TBD use his eqn 5, 6 to infer weighting coefficients?
    leapfrog_smoother = param.Number(default=0.08, bounds=(0, 0.5), doc="""
        Dampen the oscillatory instability of the leapfrog scheme by smoothing
        at each time step.
        Shampine et al. 1983 quote Bragg as suggesting 1/4 + 1/2 + 1/4 (p. 167).
        This behaviour would be recovered for leapfrog_smoother=0.5.
        Default as per pers. comm. Miles McPhee.""")

    tracer_viscosities = param.ListSelector(
        default=[], objects=['Km', 'Kh', 'Ks'], doc="""
        List of eddy viscosities, one for each tracer and in the same order.
        """)

    def step(self, **kwarg):
        """
        Step the state using a given model and forcing.
        """
        stepper = self
        forcing = stepper.forcing
        model   = stepper.model
        state   = model.state

        stepper.param.debug('======== Starting time step: {} ========'.format(state.time))

        if forcing is not None:
            # release readonly
            forcing.force(state)
            # TBD could set readonly-attributes in forcing.force
        model.adjust_model_parameters()

        # TBD could have some mechanism that prevents parameters from updating if they were already set by forcing... !
        stepper._update_state() # Need implementation in subclass

        # Step integration
        stepper._step_time()
        stepper._step_fields() # Need implementation in subclass

    def _step_time(self, **kwarg):
        self.model.state.time += self.deltat/24

    def _step_fields(self, **kwarg):
        raise NotImplementedError('Please implement in a subclass')

    def _update_state(self, **kwarg):
        raise NotImplementedError('Please implement in a subclass')

    def get_specs(self, var):
        """
        Supply specifications for time integration.

        Returns in a dict:
        ---------
            theta_jm1: Previous time step's state (np.array)
            dt2:       twice the fundamental time step (in seconds)
            dz, dzz:   Grid cell spacing
            K:         Eddy diffusivity, m^2/s (np.array)
            Q:         Local source term (np.array)
            F0:        surface flux, positive upward
            btm_bc_val: bottom boundary prescribed value
            btm_bc_kind: bottom boundary condition type
        """
        stepper = self
        model = stepper.model
        state = model.state

        # twice the fundamental time step, now in seconds:
        dt2 = 2 * stepper.deltat * 3600

        # The equations in chapter 7 are formulated for z upward positive.
        # To avoid confusion we adopt that convention in this integration:
        dz, dzz = -state.dz, -state.dzz

        if var == 'T':
            theta = state.T          # current time step, j
            theta_jm1 = state.T_previous # previous time step, j-1
            K = state.Kh
            Q = model.T_source
            F0 = state.wT0
            btm_bc_val = model.T_btm_bc
            btm_bc_kind = model.T_btm_bc_kind

        elif var == 'S':
            theta = state.S
            theta_jm1 = state.S_previous
            K = state.Ks
            Q = model.S_source
            F0 = state.wS0
            btm_bc_val  = model.S_btm_bc
            btm_bc_kind = model.S_btm_bc_kind

        elif var == 'tracer':
            raise NotImplementedError('Need to vectorize leapfrog first')
            # if (
            #     state.tracer is not None
            #     and len(model.tracer_viscosities) != state.tracer.shape[1]
            # ):
            #     raise ValueError("""Got {} viscosities but {} tracers""".format(
            #         len(model.tracer_viscosities), state.tracer.shape[1]
            #     ))

        elif var == 'U':
            theta = state.U
            theta_jm1 = state.U_previous
            K = state.Km
            if model.U_source is None:
                U_source = np.zeros(np.shape(theta))
            else:
                U_source = model.U_source
            Q = -1j*model.fcor*state.U + U_source
            F0 = -abs(state.ustar0) * state.ustar0
            btm_bc_val = 0
            btm_bc_kind = 'value'

        return {
            'dt2': dt2,
            'dz': dz,
            'dzz': dzz,
            'theta': theta,
            'theta_jm1': theta_jm1,
            'K': K,
            'Q': Q,
            'F0': F0,
            'btm_bc_val': btm_bc_val,
            'btm_bc_kind': btm_bc_kind,
        }

    def _step_field(self, var, **kwarg):
        """
        respect source function Q=Q(z), e.g. solar radiation
        """
        stepper = self
        specs = stepper.get_specs(var)

        theta = specs['theta']
        theta_jm1 = specs['theta_jm1']

        theta_jp1 = stepper._solve_leapfrog(
            theta_jm1,
            specs['dt2'],
            specs['dz'],
			specs['dzz'],
			specs['K'],
			specs['Q'],
			specs['F0'],
            specs['btm_bc_val'],
			specs['btm_bc_kind'],
        )

        # Prepare next time step

        # Use entire slices to make sure we assign back to state.T, S, ...
        # instead of creating new copy
        theta_jm1[:] = (
            theta
            + 0.5 * stepper.leapfrog_smoother * (theta_jm1 + theta_jp1 - 2*theta)
        )
        theta[:] = theta_jp1[:]


    # TBD rename into "solve_implicit" or something similar. doesn't really have
    # to do with the leap frog scheme...
    @staticmethod
    def _solve_leapfrog(theta_jm1, dt2, dz, dzz, K, Q, F0, btm_bc_val, btm_bc_kind):
        """
        Step scalar field theta using a leap-frog scheme.

        TBD
        Use an implicit scheme where vertical gradients are calculated based on
        the future (j+1) time step. This improves stability (no "overshooting")
        of flux divergences.

        Arguments
        ---------
            theta_jm1: Previous time step's state (np.array)
            dt2:       twice the fundamental time step (in seconds)
            dz, dzz:   Grid cell spacing
            K:         Eddy diffusivity, m^2/s (np.array)
            Q:         Local source term (np.array)
            F0:        surface flux, positive upward
            btm_bc_val: bottom boundary prescribed value
            btm_bc_kind: bottom boundary condition type
        """
        dtype = theta_jm1.dtype
        # init
        nz = len(theta_jm1)
        A = np.zeros(nz-1, dtype=dtype)
        B = np.zeros(nz-1, dtype=dtype)
        C = np.zeros(nz-1, dtype=dtype)
        D = np.zeros(nz-1, dtype=dtype)
        E = np.zeros(nz-1, dtype=dtype)

        if Q is None:
            Q = np.zeros(np.shape(theta_jm1))

        # Eq. 7.5
        A[0] = -dt2 * K[1]/(dz[0]*dzz[0])
        ix = slice(1, nz-1)
        ix_plus_1 = slice(2, nz)
        ix_minus_1 = slice(0, nz-2)
        A[ix] = -dt2 * K[ix_plus_1] / (dz[ix] * dzz[ix])
        C[ix] = -dt2 * K[ix] / (dz[ix] * dzz[ix_minus_1])
        B[ix] = 1 - A[ix] - C[ix]

        # initialize updated scalar
        theta = np.zeros(nz, dtype=dtype)

        # surface flux condition
        # Eq. 7.12.
        D[0] = (dt2* (F0/dz[0] + Q[0]) + theta_jm1[0]) / (1 - A[0])
        E[0] = A[0]/(A[0] - 1)

        # Eq. 7.8
        for i in range(1, nz-1):
            den = B[i]+C[i]*E[i-1]
            D[i] = (theta_jm1[i] + dt2*Q[i] - C[i]*D[i-1]) / den
            E[i] = -A[i] / den


        # strictly speaking, theta[-1] is not part of the domain, but we might
        # as well avoid theta[-1] = NaN
        if btm_bc_kind == 'flux':
            Fbtm = -btm_bc_val # z upward positive
            # Eq. 7.24, but moved one index down
            theta[-1] = (
                dt2 * (Q[-1] + Fbtm / dz[-1])
                + theta_jm1[-1]
                - C[-1] * D[-2]
            ) / (1 + C[-1] * (E[-2] - 1))
        elif btm_bc_kind == 'value':
            theta[-1] = btm_bc_val
            theta[-2] = D[-2] + E[-2] * theta[-1]
        elif btm_bc_kind == 'fixed':
            theta[-1] = theta_jm1[-1]
            theta[-2] = D[-2] + E[-2] * theta[-1]

        # Eq. 7.6
        for i in range(nz-2, -1, -1):
            theta[i] = D[i] + E[i] * theta[i+1]

        return theta

    @staticmethod
    def _solve_leapfrog_coriolis(theta_jm1, dt2, fcor, dz, dzz, K, Q, F0):
        """
        This code only left here for educational purposes. It is not supposed to be
        used as it treats the Coriolis force not as a time-centered source but as
        part of the previous time step.

        Solve leapfrog specifically for dU/dt = dtau/dz - ifU (using approach as
        described in McPhee 2008, Ch. 7), treating ifU not as a source, but
        absorbing it into the difference scheme.
        """
        # init
        nz = len(theta_jm1)
        A = np.zeros(nz-1, dtype=complex)
        B = np.zeros(nz-1, dtype=complex)
        C = np.zeros(nz-1, dtype=complex)
        D = np.zeros(nz-1, dtype=complex)
        E = np.zeros(nz-1, dtype=complex)

        if Q is None:
            Q = np.zeros(np.size(theta_jm1))

        # Eq. 7.5
        A[0] = -dt2 * K[1]/(dz[0]*dzz[0])
        ix = slice(1, nz-1)
        ix_plus_1 = slice(2, nz)
        ix_minus_1 = slice(0, nz-2)
        A[ix] = -dt2 * K[ix_plus_1] / (dz[ix] * dzz[ix])
        C[ix] = -dt2 * K[ix] / (dz[ix] * dzz[ix_minus_1])
        B[ix] = 1 + 1j*dt2*fcor - A[ix] - C[ix] # This line changed from scalar case

        # surface flux condition
        # changed from scalar case
        denominator = (1 - A[0] + 1j*dt2*fcor)
        D[0] = (dt2* (F0/dz[0] + Q[0]) + theta_jm1[0]) / denominator
        E[0] = - A[0] / denominator

        # Eq. 7.8 (unchanged)
        for i in range(1, nz-1):
            denominator = B[i]+C[i]*E[i-1]
            D[i] = (theta_jm1[i] + dt2*Q[i] - C[i]*D[i-1]) / denominator
            E[i] = -A[i] / denominator

        # initialize updated scalar
        theta = np.zeros(nz, dtype=complex)

        for i in range(nz-2, -1, -1):
            theta[i] = D[i] + E[i] * theta[i+1]

        return theta


class TempSalStepper(Stepper):
    """
    Step T, S fields using the steady LTC solution at each time step.
    """

    def _step_fields(self, **kwarg):
        stepper = self
        stepper._step_field(var='T', **kwarg)
        stepper._step_field(var='S', **kwarg)

    def _update_state(self, **kwarg):
        stepper = self
        model = stepper.model
        state = model.state
        state.update_ctd()
        model.update_interface()
        model.find_steady_k() # new Km (and current field)


class TempSalVelStepper(Stepper):
    """
    Step T, S, u, v fields using the full momentum equation.
    """

    def _step_fields(self, **kwarg):
        stepper = self
        stepper._step_field(var='U', **kwarg)
        stepper._step_field(var='T', **kwarg)
        stepper._step_field(var='S', **kwarg)

    def _update_state(self, **kwarg):
        stepper = self
        model = stepper.model
        state = model.state
        state.update_ctd()
        model.update_interface()
        model.update_turbulence()
