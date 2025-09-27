import numpy as np
import control as ct

m_1 = 1.0  # mass of the wheel (kg)
m_2 = 2.0  # mass of the body (kg)
i_1 = 0.006  # inertia of the wheel (kg.m^2)
i_2 = 0.025  # inertia of the body (kg.m^2
l = 0.5  # length to the center of mass of the body (m)
g = 9.81  # acceleration due to gravity (m/s^2)
r = 0.1  # radius of the wheel (m)
l = 1.0

z_1 = -l * (m_1 * r + i_1 / r)
z_2 = g * ((m_1 + m_2) * r + i_1 / r)
z_3 = m_1 * r**2 + i_1
z_4 = m_2 * g * r

A = [
    [0,     1,    0,           0],
    [0,     0,    z_4 / z_3,   0],
    [0,     0,    0,           1],
    [0,     0,    z_2 / z_1,   0]
]

B = [
    [0],
    [1 / (m_1 * r**2 + i_1)],
    [0],
    [-1 / (m_1 * r + (i_1 / r))]
]

Q = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

R = [
    [0.1]
]

K, S, E = ct.lqr(A, B, Q, R)

print(K)