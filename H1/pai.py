import random

def estimate_pi(num_samples=1000000):
    inside_circle = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

print("Estimated π:", estimate_pi(1000000))