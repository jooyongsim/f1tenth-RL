import matplotlib.pyplot as plt
import csv

episodes, rewards = [], []

with open("episode_rewards_sanitycheck.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward"]))

plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.show()
