import matplotlib.pyplot as plt
import csv

steps = []
losses = []

with open("loss_log_normalization_trained.csv", mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        losses.append(float(row["loss"]))

plt.plot(steps, losses, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.grid(True)
plt.legend()
plt.show()
