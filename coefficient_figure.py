import numpy as np
import matplotlib.pyplot as plt

def ns_scalar_step(x, a, b, c):
    """One scalar Newton-Schulz iteration."""
    return a * x + b * x**3 + c * x**5

def hybrid_ns_scalar(x, k):
    """
    Apply k steps of the hybrid Newton-Schulz iteration.

    First 8 steps:  (a,b,c) = (3.4445, -4.7750, 2.0315)
    Final 2 steps:  (a,b,c) = (2.0, -1.5, 0.5)
    """
    y = x.copy()

    for step in range(1, k + 1):
        if step <= 8:
            a, b, c = 3.4445, -4.7750, 2.0315
        else:
            a, b, c = 2.0, -1.5, 0.5

        y = ns_scalar_step(y, a, b, c)

    return y

# Input singular values are typically normalized to lie in [0, 1]
x = np.linspace(0, 1, 1000)

plt.figure(figsize=(7, 5))

for k in [2, 4, 6, 8, 10]:
    y = hybrid_ns_scalar(x, k)
    plt.plot(x, y, label=f"k = {k}")

plt.plot(x, x, "--", linewidth=1, label="identity")

plt.xlabel("Input Singular value")
plt.ylabel("Output Singular value")
plt.title("Hybrid Newton-Schulz Iteration")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("muon.png", dpi=300, bbox_inches="tight", pad_inches=0)
