import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
from scipy import integrate
from tqdm import tqdm

from plotConcentration import C, C1

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


def case_2():
    """Create effectiveness matrix for values of d and delta_t in the case where active duration defines quality. Runs at roughly 4-6it/s."""
    times = np.linspace(0, TOTAL_HOURS, T_PRECISION)

    d_range = np.linspace(MIN_DOSAGE, MAX_DOSAGE, NUM_POINTS)
    delta_range = np.linspace(1, TOTAL_HOURS, NUM_POINTS)

    e_matrix = np.zeros((NUM_POINTS, NUM_POINTS))

    best_treatment = None
    max_effectiveness = 0

    with tqdm(total=NUM_POINTS * NUM_POINTS) as bar:
        for i, d in enumerate(d_range):
            for j, delta_t in enumerate(delta_range):
                T = np.arange(0, TOTAL_HOURS, delta_t)
                c = C(times, T=T, d=d, p=p, r=r)
                if sum(map(lambda x: x > MTC, c)) > 0:
                    e_matrix[i, j] = 0
                    bar.update(1)
                    continue
                e_matrix[i, j] = sum(map(lambda x: x >= MEC and x <= MTC, c))
                if e_matrix[i, j] > max_effectiveness:
                    max_effectiveness = e_matrix[i, j]
                    best_treatment = {
                        'Dosage': d,
                        'Delta t': delta_t,
                        'Effectiveness': max_effectiveness
                    }
                bar.update(1)

    print(best_treatment)
    return best_treatment


def case_1():
    """Create effectiveness matrix for values of d and delta_t in the case where active area defines quality. Runs at roughly 2-3it/s."""

    times = np.linspace(0, TOTAL_HOURS, T_PRECISION)

    d_range = np.linspace(5, 500, NUM_POINTS)
    delta_range = np.linspace(1, TOTAL_HOURS, NUM_POINTS)

    e_matrix = np.zeros((NUM_POINTS, NUM_POINTS))

    best_treatment = None
    max_effectiveness = 0

    with tqdm(total=NUM_POINTS * NUM_POINTS) as bar:
        for i, d in enumerate(d_range):
            for j, delta_t in enumerate(delta_range):
                T = np.arange(0, TOTAL_HOURS, delta_t)
                c = C(times, T=T, d=d, p=p, r=r)

                if d == 500 and delta_t == 1:
                    print(c)

                # if C ever exceeds MTC, throw out the treatment plan
                if sum(map(lambda x: x > MTC, c)) > 0:
                    e_matrix[i, j] = 0
                    bar.update(1)
                    continue

                roots = []
                # first we find all roots of C(t) - MEC
                # we can do so by finding wherever the sign changes
                # if we miss a sign change due to precision, the integral will not differ by much
                # if a root does not result in sign change (touches axis), it has no effect on integral
                sgn = np.sign(c - MEC)
                for k in range(len(sgn) - 1):
                    if sgn[k] + sgn[k + 1] == 0:
                        roots.append(times[k])

                area = 0
                # next we integrate over every pair of roots
                # the first root is always the point where C goes above the MEC
                for k in range(len(roots)):
                    if k % 2 == 1:
                        continue
                    else:
                        # integrate from roots[k] to roots[k+1]
                        if k + 1 < len(roots):
                            I, _ = integrate.quad(C1, roots[k],
                                                  roots[k + 1], args=(T, p, d, r))
                        else:
                            I, _ = integrate.quad(C1,
                                                  roots[k], TOTAL_HOURS, args=(T, p, d, r))
                        area += I

                e_matrix[i, j] = area

                if e_matrix[i, j] > max_effectiveness:
                    max_effectiveness = e_matrix[i, j]
                    best_treatment = {
                        'Dosage': d,
                        'Delta t': delta_t,
                        'Effectiveness': max_effectiveness
                    }

                bar.update(1)

    print(best_treatment)
    return best_treatment


if __name__ == '__main__':
    if sys.argv[1] == '1':
        case_1()
    elif sys.argv[1] == '2':
        case_2()
    else:
        print('Invalid argument.')
