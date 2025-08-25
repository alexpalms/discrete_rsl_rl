# Read csv files and plot the results

import pandas as pd
import matplotlib.pyplot as plt

# Continuous PPO
# Read csv files
df = pd.read_csv("runs/continuous_PPO.csv")

# Plot the results
plt.figure(figsize=(10,6))
font = {'size': 14}
plt.rc('font', **font)
# Add horizonal line at 10
plt.axhline(y=10, color="black", linestyle="--", label="Target Reward")
plt.plot(df["Relative RSL"], df["Value RSL"], label="RSL")
plt.plot(df["Relative SB3"], df["Value SB3"], label="SB3")
plt.legend(fontsize=16)
plt.ylabel("Cumulative Reward", fontsize=14, labelpad=10)
plt.xlabel("Wall Time [s]", fontsize=14, labelpad=10)
plt.title("Legged Locomotion - Continuous PPO Training", fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Multidiscrete PPO
# Read csv files
df = pd.read_csv("runs/multidiscrete_PPO.csv")

# Plot the results
plt.figure(figsize=(10,6))
font = {'size': 14}
plt.rc('font', **font)
# Add horizonal line at 0.5
plt.axhline(y=0.5, color="black", linestyle="--", label="Target Failure Risk")
plt.plot(df["Relative RSL"], -df["Value RSL"], label="RSL")
plt.plot(df["Relative SB3"], -df["Value SB3"], label="SB3")
plt.legend(fontsize=16)
plt.ylabel("Failure Risk Probability []", fontsize=14, labelpad=10)
plt.xlabel("Wall Time [s]", fontsize=14, labelpad=10)
plt.title("Maintenance Scheduling - Multidiscrete PPO Training", fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()