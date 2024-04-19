from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import append, arange, arctan, array, concatenate, cos, reshape, sin, zeros

#from core.dynamics import FBLinDynamics, RoboticDynamics, SystemDynamics

from dynamics.fb_lin_dynamics import FBLinDynamics
from dynamics.system_dynamics import SystemDynamics

from dynamics.robotic_dynamics import RoboticDynamics
from dynamics.util import default_fig
import logging


class PlanarQuadrotorForceInput(RoboticDynamics):
    def __init__(self, m, J, b, g=9.81):
        '''Create a PlanarQuadrotorForceInput object that satisfies the RoboticDynamics interface.'''
        try:
            super().__init__(3, 2)
            self.params = m, J, b, g
            self.standardizer_u = None
            self.standardizer_x = None
            logging.info("PlanarQuadrotorForceInput initialized with m=%s, J=%s, b=%s, g=%s", m, J, b, g)
        except Exception as e:
            logging.error("Failed to initialize PlanarQuadrotorForceInput: %s", e)
            raise

    def D(self, q):
        '''Compute positive-definite inertia matrix.'''
        try:
            m, J, b, _ = self.params
            inertia_matrix = array([[m, 0, 0], [0, m, 0], [0, 0, J/b]])
            logging.debug("Inertia matrix computed: %s", inertia_matrix)
            return inertia_matrix
        except Exception as e:
            logging.error("Failed to compute inertia matrix: %s", e)
            raise

    def C(self, q, q_dot):
        '''Compute Coriolis terms.'''
        try:
            coriolis_matrix = zeros((3, 3))
            logging.debug("Coriolis matrix computed: %s", coriolis_matrix)
            return coriolis_matrix
        except Exception as e:
            logging.error("Failed to compute Coriolis matrix: %s", e)
            raise

    def U(self, q):
        '''Compute potential energy.'''
        try:
            m, _, _, g = self.params
            _, z, _ = q
            potential_energy = m * g * z
            logging.debug("Potential energy computed: %s", potential_energy)
            return potential_energy
        except Exception as e:
            logging.error("Failed to compute potential energy: %s", e)
            raise

    def G(self, q):
        '''Compute conservative forces.'''
        try:
            m, _, _, g = self.params
            conservative_forces = array([0, m * g, 0])
            logging.debug("Conservative forces computed: %s", conservative_forces)
            return conservative_forces
        except Exception as e:
            logging.error("Failed to compute conservative forces: %s", e)
            raise

    def B(self, q):
        '''Compute actuation terms.'''
        try:
            _, _, theta = q
            actuation_matrix = array([[-sin(theta), -sin(theta)], [cos(theta), cos(theta)], [-1, 1]])
            logging.debug("Actuation matrix computed: %s", actuation_matrix)
            return actuation_matrix
        except Exception as e:
            logging.error("Failed to compute actuation matrix: %s", e)
            raise


    def plot_states(self, ts, xs, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        ax.set_title('States', fontsize=16)
        ax.set_xlabel('$q$', fontsize=16)
        ax.set_ylabel('$\\dot{q}$', fontsize=16)
        ax.plot(xs[:, 0], xs[:, 3], linewidth=3, label='$x$ (m)')
        ax.plot(xs[:, 1], xs[:, 4], linewidth=3, label='$z$ (m)')
        ax.plot(xs[:, 2], xs[:, 5], linewidth=3, label='$\\theta$ (rad)')
        ax.legend(fontsize=16)

        return fig, ax

    def plot_actions(self, ts, us, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        if labels is None:
            labels = ['$f$ (N)', '$\\tau$ (N $\\cdot$ m)']

        ax.set_title('Actions', fontsize=16)
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)
        ax.plot(*us.T, linewidth=3)

        return fig, ax

    def plot_tangents(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        ax.set_title('Tangent Vectors', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.plot(*xs[:, :2].T, linewidth=3)
        ax.quiver(*xs[::skip, :2].T, *xs[::skip, 3:5].T, angles='xy')

        return fig, ax

    def plot_physical(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        xs, zs, thetas = xs[:, :3].T
        dirs = array([sin(thetas), cos(thetas)])[:, ::skip]

        ax.set_title('Physical Space', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.quiver(xs[::skip], zs[::skip], *dirs, angles='xy')
        ax.plot(xs, zs, linewidth=3)
        ax.axis('equal')

        return fig, ax

    def plot(self, xs, us, ts, fig=None, action_labels=None, skip=1):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        physical_ax = fig.add_subplot(1, 2, 1)
        fig, physical_ax = self.plot_physical(ts, xs, fig, physical_ax, skip)

        action_ax = fig.add_subplot(1, 2, 2)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax, action_labels)

        return fig, (physical_ax, action_ax)
