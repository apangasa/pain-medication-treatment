import math

import matplotlib.pyplot as plt
import numpy as np


MEC = 80
MTC = 200
TOTAL_HOURS = 7 * 24

# Precision
INCREMENT = None
NUM_POINTS = 20

if INCREMENT:
    # computes how many values of d and delta_t will be tested
    NUM_POINTS = TOTAL_HOURS / INCREMENT

T_PRECISION = 5000  # how many times along the t-axis will be plugged into the function

# Situation Bounds
MIN_DOSAGE = 5
MAX_DOSAGE = 500

# Parameters
p = 1  # proportion of usable drug
r = 0.25  # rate of drug absorption into the blood

d = 187  # dosage
delta_t = 14.6  # every delta_t hours, the patient takes d milligrams of the drug

T = np.arange(0, TOTAL_HOURS, delta_t)  # list of times at which drug is taken


def C_i1(t, d=d, p=p, r=r):
    return max(p * d * (1 - (math.e ** (-1 * r * t))) * (math.e ** (-0.06 * t)), 0)


def C1(t, T=T, d=d, p=p, r=r):
    c = 0
    for t_i in T:
        c += C_i1(t - t_i, d=d, p=p, r=r)
    return c


def C_i(times, d=d, p=p, r=r):
    """Computes blood concentration at each time in times as contribution from dose i."""
    return np.array([max(p * d * (1 - (math.e ** (-1 * r * t))) * (math.e ** (-0.06 * t)), 0) for t in times])


def C(times, T=T, d=d, p=p, r=r):
    """Compute aggregate blood concentration at each time in times."""
    c = np.zeros_like(times)
    for t_i in T:
        c = np.add(c, C_i([t - t_i for t in times], d=d, p=p, r=r))
    return c


if __name__ == '__main__':
    # consider 5000 points across the first 100 hours
    times = np.linspace(0, TOTAL_HOURS, 5000)
    concs = C(times)
    mec_line = MEC * np.ones(T_PRECISION)
    mtc_line = MTC * np.ones(T_PRECISION)

    plt.plot(times, concs, label='Concentration')
    plt.plot(times, mec_line, linestyle='--', label='MEC')
    plt.plot(times, mtc_line, linestyle='--', label='MTC')
    plt.legend()
    plt.show()
