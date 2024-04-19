from numpy import dot, zeros, array
from dynamics.affine_dynamics import AffineDynamics
from dynamics.linearizable_dynamics import LinearizableDynamics
from dynamics.system_dynamics import SystemDynamics
import logging

class LinearLiftedDynamics(SystemDynamics, AffineDynamics, LinearizableDynamics):
    def __init__(self, A, B, C, basis, continuous_mdl=True, dt=None, standardizer_x=None, standardizer_u=None):
        logging.debug("Initializing LinearLiftedDynamics")
        try:
            if B is not None:
                n, m = B.shape
            else:
                n, m = A.shape[0], None

            assert A.shape == (n, n), "Shape of A must be (n, n)"

            super().__init__(n, m)
            self.A = A
            self.B = B
            self.C = C
            self.basis = basis
            self.continuous_mdl = continuous_mdl
            self.dt = dt
            self.standardizer_x = standardizer_x
            self.standardizer_u = standardizer_u
            logging.info(f"LinearLiftedDynamics successfully initialized with continuous model: {continuous_mdl}")
        except Exception as e:
            logging.error(f"Failed to initialize LinearLiftedDynamics: {e}")
            raise

    def drift(self, x, t):
        try:
            result = np.dot(self.A, x)
            logging.debug(f"Computed drift at time {t} for state {x}: {result}")
            return result
        except Exception as e:
            logging.error(f"Error computing drift for x={x}, t={t}: {e}")
            raise

    def act(self, x, t):
        try:
            logging.debug(f"Returning actuation matrix at time {t} for state {x}")
            return self.B
        except Exception as e:
            logging.error(f"Error accessing actuation matrix for x={x}, t={t}: {e}")
            raise

    def eval_dot(self, x, u, t):
        try:
            if self.B is None:
                return self.drift(x, t)
            else:
                result = np.dot(self.A, x) + np.dot(self.B, u)
                logging.debug(f"Evaluated dot product at time {t} for state {x} and input {u}: {result}")
                return result
        except Exception as e:
            logging.error(f"Error evaluating dynamics for x={x}, u={u}, t={t}: {e}")
            raise

    def lift(self, x, u):
        try:
            lifted = self.basis(x)
            logging.debug(f"Lifted state {x} to {lifted} using input {u}")
            return lifted
        except Exception as e:
            logging.error(f"Error lifting state {x} with input {u}: {e}")
            raise

    def simulate(self, x_0, controller, ts, processed=True, atol=1e-6, rtol=1e-6):
        """
        Simulate the dynamics.
        ] x_0: Initial state vector, numpy array
        - controller: Controller object
        - ts: Time vector, numpy array
        - processed: Flag indicating if the inputs are processed, bool
        - atol: Absolute tolerance for the simulation, float
        - rtol: Relative tolerance for the simulation, float
        Returns:
        - The simulated state trajectory and input trajectory, tuple
        """
        logging.info("Starting simulation.")
        try:
            if self.continuous_mdl:
                logging.debug("Running continuous model simulation.")
                xs, us = SystemDynamics.simulate(self, x_0, controller, ts, processed=processed, atol=atol, rtol=rtol)
            else:
                N = len(ts)
                xs = np.zeros((N, self.n))
                us = [None] * (N - 1)

                controller.reset()

                xs[0] = x_0
                logging.debug(f"Initial state set at x_0: {x_0}")
                for j in range(N - 1):
                    x = xs[j]
                    t = ts[j]
                    u = controller.eval(x, t)
                    us[j] = u
                    logging.debug(f"Evaluated control at time {t}: {u}")
                    u = controller.process(u)
                    xs[j + 1] = self.eval_dot(x, u, t)
                    logging.debug(f"State at time {ts[j + 1]}: {xs[j + 1]}")
                if processed:
                    us = np.array([controller.process(u) for u in us])
                    logging.debug("Processed control inputs after simulation.")

            logging.info("Simulation completed successfully.")
            return xs, us
        except Exception as e:
            logging.error(f"Simulation failed: {e}")
            raise
