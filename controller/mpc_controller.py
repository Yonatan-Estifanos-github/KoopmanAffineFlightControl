import numpy as np
import scipy.sparse as sparse
from scipy.signal import cont2discrete
import osqp
from controller.controller import Controller
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MPCController(Controller):
    """
    Class for linear MPC with lifted linear dynamics.
    Quadratic programs are solved using OSQP.
    """
    def __init__(self, lifted_linear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, q_d, const_offset=None, terminal_constraint=False, add_slack=False):
        logging.debug("Initializing MPCController")
        Controller.__init__(self, lifted_linear_dynamics)
        self.dynamics_object = lifted_linear_dynamics
        self.dt = dt
        if lifted_linear_dynamics.continuous_mdl:
            Ac = lifted_linear_dynamics.A
            Bc = lifted_linear_dynamics.B
            [self.nx, self.nu] = Bc.shape
            lin_model_d = cont2discrete((Ac, Bc, np.eye(self.nx), np.zeros((self.nu, 1))), dt)
            self._osqp_Ad = sparse.csc_matrix(lin_model_d[0])
            self._osqp_Bd = sparse.csc_matrix(lin_model_d[1])
            logging.debug("Converted continuous dynamics to discrete")
        else:
            self._osqp_Ad = lifted_linear_dynamics.A
            self._osqp_Bd = lifted_linear_dynamics.B
            [self.nx, self.nu] = self._osqp_Bd.shape
        
        self.C = lifted_linear_dynamics.C
        self.Q = Q
        self.QN = QN
        self.R = R
        self.N = N
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.const_offset = const_offset if const_offset is not None else np.zeros(self.nu)
        self.q_d = q_d
        self.ns = q_d.shape[0]
        if self.q_d.ndim == 2:
            self.q_d = np.hstack([self.q_d, np.transpose(np.tile(self.q_d[:, -1], (self.N + 1, 1)))])
        self.terminal_constraint = terminal_constraint
        self.add_slack = add_slack

        self.build_objective_()
        self.build_constraints_()
        self.prob = osqp.OSQP()
        self.prob.setup(self._osqp_P, self._osqp_q, self._osqp_A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)
        self._osqp_result = None
        self.comp_time = []
        logging.debug("MPCController initialized")

    def build_objective_(self):
        logging.debug("Building objective function")
        CQC = sparse.csc_matrix(np.transpose(self.C).dot(self.Q.dot(self.C)))
        CQNC = sparse.csc_matrix(np.transpose(self.C).dot(self.QN.dot(self.C)))
        Q_slack = 1e3 * sparse.eye(self.ns * (self.N + 1))

        if not self.add_slack:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), CQC), CQNC,
                                              sparse.kron(sparse.eye(self.N), self.R)]).tocsc()
        else:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), CQC), CQNC,
                                              sparse.kron(sparse.eye(self.N), self.R),
                                              Q_slack]).tocsc()
        logging.debug("Objective function built")

    def build_constraints_(self):
        logging.debug("Building constraints")
        x0 = np.zeros(self.nx)
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self._osqp_Bd)

        if not self.add_slack:
            Aineq = sparse.block_diag([self.C for _ in range(self.N + 1)] + [np.eye(self.N * self.nu)])
            Aeq = sparse.hstack([Ax, Bu])
        else:
            Aineq = sparse.hstack([sparse.block_diag([self.C for _ in range(self.N + 1)] + [np.eye(self.N * self.nu)]),
                                   sparse.vstack([sparse.eye(self.ns * (self.N + 1)),
                                                  sparse.csc_matrix((self.nu * self.N, self.ns * (self.N + 1)))]).tocsc()])
            Aeq = sparse.hstack([Ax, Bu, sparse.csc_matrix((self.nx * (self.N + 1), self.ns * (self.N + 1)))])
        self._osqp_A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_l = np.hstack([x0, np.zeros(self.N * self.nx)])
        self._osqp_u = np.hstack([x0, np.zeros(self.N * self.nx)])
        logging.debug("Constraints built")

    def eval(self, x, t):
        logging.debug(f"Evaluating control at time {t}")
        x = self.dynamics_object.lift(x.reshape((1, -1)), None).squeeze()
        self._osqp_l[:self.nx] = -x
        self._osqp_u[:self.nx] = -x
        self.prob.update(l=self._osqp_l, u=self._osqp_u)
        self._osqp_result = self.prob.solve()
        self.comp_time.append(self._osqp_result.info.run_time)
        logging.debug("Control evaluated")
        return self._osqp_result.x[self.nx * (self.N + 1):self.nx * (self.N + 1) + self.nu]

    def parse_result(self):
        logging.debug("Parsing result")
        return np.transpose(np.reshape(self._osqp_result.x[:(self.N + 1) * self.nx], (self.N + 1, self.nx)))

    def get_control_prediction(self):
        logging.debug("Getting control prediction")
        return np.transpose(np.reshape(self._osqp_result.x[self.nx * (self.N + 1):self.nx * (self.N + 1) + self.nu * self.N], (self.N, self.nu)))
