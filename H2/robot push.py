# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Transition matrix T_push[x_next, x_current]
T_push = np.array([[1.0, 0.6],
                   [0.0, 0.4]])

# Measurement likelihood M[z, x] with z ∈ {open(0), closed(1)}
M = np.array([[0.9, 0.5],   # z=open likelihoods for (open, closed)
              [0.1, 0.5]])  # z=closed

z_seq_labels = ["open","open","open","open","open","closed","open","open"]
z_index = {"open": 0, "closed": 1}
z_seq = [z_index[z] for z in z_seq_labels]

def predict(bel, T):
    """bel_bar(x') = sum_x P(x'|x,u) * bel(x) = T @ bel"""
    return T @ bel

def update(bel_bar, M, z):
    """bel_plus(x) ∝ P(z|x) * bel_bar(x)"""
    unnorm = M[z] * bel_bar
    eta = 1.0 / np.sum(unnorm)
    bel_plus = eta * unnorm
    return bel_plus, eta, unnorm

bel = np.array([0.5, 0.5])  # [P(open), P(closed)]

rows = []
print("{:>4s}  {:>6s} {:>6s}   {:>6s} {:>6s}   {:>6s} {:>6s}   {:>6s}".format(
    "step", "bel_o", "bel_c", "bbar_o", "bbar_c", "z", "b+_o", "b+_c"))
print("-"*68)
print("{:>4d}  {:6.3f} {:6.3f}   {:>6s} {:>6s}   {:>6s} {:6.3f} {:6.3f}".format(
    0, bel[0], bel[1], "", "", "", bel[0], bel[1]))

bel_plus_open_curve = [bel[0]]  # include step 0 for plotting (optional)
steps = [0]

for i, z in enumerate(z_seq, start=1):
    bel_bar = predict(bel, T_push)

    bel_plus, eta, unnorm = update(bel_bar, M, z)

    print("{:>4d}  {:6.3f} {:6.3f}   {:6.3f} {:6.3f}   {:>6s} {:6.3f} {:6.3f}".format(
        i, bel[0], bel[1], bel_bar[0], bel_bar[1],
        "open" if z == 0 else "closed", bel_plus[0], bel_plus[1]))

    steps.append(i)
    bel_plus_open_curve.append(bel_plus[0])

    bel = bel_plus

# ---------- Plot ----------
plt.figure()
plt.plot(steps, bel_plus_open_curve, marker='o', label='bel⁺(open)')
plt.xlabel("Step")
plt.ylabel("Probability")
plt.title("Belief (Open) over Steps — Push + Measurement Sequence")
plt.ylim(0.0, 1.05)
plt.grid(True)
plt.legend()
plt.show()