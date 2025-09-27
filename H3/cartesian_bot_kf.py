#!/usr/bin/env python

import numpy
import math

default_var = 0.25
default_time_step = 1 / 50

class CartesianBotKF:
    def __init__(self, torque_const, rotor_inertia, friction_const,
                 armature_res, armature_ind, wheel_radius):
        self.K = torque_const
        self.J = rotor_inertia
        self.b = friction_const
        self.R = armature_res
        self.L = armature_ind
        self.r = wheel_radius
        self.state = numpy.zeros((6, 1))
        self.state_cov = numpy.identity(6) * default_var
        self.predicted_state = numpy.zeros((6, 1))
        self.predicted_state_cov = numpy.identity(6) * default_var
        self.time = None

    def set_state(self, timestamp,
                  position_x = 0, position_y = 0,
                  wheel_ang_vel_x = 0, wheel_ang_vel_y = 0,
                  motor_current_x = 0, motor_current_y = 0):
        self.time = timestamp
        self.state[0, 0] = position_x
        self.state[1, 0] = wheel_ang_vel_x
        self.state[2, 0] = motor_current_x
        self.state[3, 0] = position_y
        self.state[4, 0] = wheel_ang_vel_y
        self.state[5, 0] = motor_current_y
        self.state_cov = numpy.identity(6) * default_var

    def A_matrix(self, delta_t = default_time_step):
        A = numpy.identity(6)
        A[0, 1] = A[3, 4] = self.r * delta_t
        A[1, 1] = A[4, 4] = 1 - delta_t * self.b / self.J
        A[1, 2] = A[4, 5] = delta_t * self.K / self.J
        A[2, 1] = A[5, 4] = -delta_t * self.K / self.L
        A[2, 2] = A[5, 5] = 1 - delta_t * self.R / self.L
        return A

    def B_matrix(self, delta_t = default_time_step):
        B = numpy.zeros((6, 2))
        B[2, 0] = B[5, 1] = delta_t / self.L
        return B

    def C_matrix(self):
        C = numpy.zeros((2,6))
        C[0, 0] = 1
        C[1, 3] = 1
        return C

    def _predict(self, timestamp, v_x, v_y,
                 var_vx = default_var, var_vy = default_var):
        if self.time is None:
            raise RuntimeError('uninitialized filter')
        delta_t = timestamp - self.time
        self.time = timestamp
        u = numpy.array([[ v_x ],
                         [ v_y ]])
        sigma_u = numpy.array([[var_vx, 0],
                               [0, var_vy]])
        self.predicted_state = self.A_matrix(delta_t) @ \
            self.state + self.B_matrix(delta_t) @ u
        self.predicted_state_cov = \
            self.A_matrix(delta_t) @ self.state_cov @ \
            self.A_matrix(delta_t).transpose() + \
            self.B_matrix(delta_t) @ sigma_u @ \
            self.B_matrix(delta_t).transpose()

    def _skip_measure(self):
        self.state = self.predicted_state
        self.state_cov = numpy.identity(6) * default_var

    def peek_pos(self):
        return self.state[0, 0], self.state[3, 0]

    def peek_omega(self):
        return self.state[1, 0], self.state[4, 0]

    def peek_current(self):
        return self.state[2, 0], self.state[5, 0]

    def _measure(self, z_x, z_y,
                 var_zx = default_var, var_zy = default_var,
                 rho_zxy = 0):
        sigma_z = numpy.array([
            [var_zx, rho_zxy * math.sqrt(var_zx) * math.sqrt(var_zy)],
            [rho_zxy * math.sqrt(var_zx) * math.sqrt(var_zy), var_zy]])
        z = numpy.array([[z_x],
                         [z_y]])
        SigmaXCT =  self.predicted_state_cov @ self.C_matrix().transpose()
        CSigmaXCT = self.C_matrix() @ SigmaXCT
        K = SigmaXCT @ numpy.linalg.inv(CSigmaXCT + sigma_z)
        self.state = self.predicted_state + K @ \
            (z - self.C_matrix() @ self.predicted_state)
        self.state_cov = (numpy.identity(6) - K @ self.C_matrix()) @ \
            self.predicted_state_cov

    def advance_filter(self, timestamp, v_x, v_y, z_x, z_y,
                       var_vx = default_var, var_vy = default_var,
                       var_zx = default_var, var_zy = default_var,
                       rho_zxy = 0):
        self._predict(timestamp, v_x, v_y, var_vx, var_vy)
        self._measure(z_x, z_y, var_zx, var_zy, rho_zxy)

    def simulate_system(self, timestamp, v_x, v_y):
        self._predict(timestamp, v_x, v_y)
        self._skip_measure()

    def get_estimate(self):
        z = self.C_matrix() @ self.state
        z_cov = self.C_matrix() @ self.state_cov @ self.C_matrix().transpose()
        return z, z_cov

    def get_state(self):
        return self.state, self.state_cov
