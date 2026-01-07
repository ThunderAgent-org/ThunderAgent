import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [15, 6]

# Load data
try:
    df = pd.read_csv('system_monitor.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: system_monitor.csv not found. Please ensure the file is in the same directory.")
    exit()

# Convert Time column to datetime objects
# The date part defaults to today/1900-01-01 depending on pandas version if only time is present, 
# but for visualization relative time or just time component matters.
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

# 1. CPU Usage and Memory
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('CPU Utilization (%)', color=color)
ax1.plot(df['Time'], df['CPU_Util(%)'], color=color, label='CPU Util')
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('CPU Memory (GiB)', color=color)
ax2.plot(df['Time'], df['CPU_Mem(GiB)'], color=color, label='CPU Mem')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('CPU Utilization and Memory Usage Over Time')
fig.tight_layout()
plt.savefig('cpu_usage.png')
print("Saved cpu_usage.png")
# plt.show() # Uncomment if running in an environment with display

# 2. System Load
plt.figure(figsize=(15, 6))
plt.plot(df['Time'], df['Load'], label='System Load', color='purple')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('System Load Over Time')
plt.legend()
plt.savefig('system_load.png')
print("Saved system_load.png")

# 3. GPU Utilization
gpu_util_cols = [col for col in df.columns if 'GPU' in col and 'Util' in col]

plt.figure(figsize=(15, 8))
for col in gpu_util_cols:
    plt.plot(df['Time'], df[col], label=col, alpha=0.7)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xlabel('Time')
plt.ylabel('Utilization (%)')
plt.title('GPU Utilization Over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('gpu_utilization.png')
print("Saved gpu_utilization.png")

# 4. GPU Memory Usage
gpu_mem_cols = [col for col in df.columns if 'GPU' in col and 'Mem' in col]

plt.figure(figsize=(15, 8))
for col in gpu_mem_cols:
    plt.plot(df['Time'], df[col], label=col, alpha=0.7)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xlabel('Time')
plt.ylabel('Memory Usage (MiB)')
plt.title('GPU Memory Usage Over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('gpu_memory.png')
print("Saved gpu_memory.png")

# 5. Disk Usage Overlay
if 'Disk_Overlay(GiB)' in df.columns:
    plt.figure(figsize=(15, 6))
    plt.plot(df['Time'], df['Disk_Overlay(GiB)'], label='Disk Overlay', color='brown')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xlabel('Time')
    plt.ylabel('Disk Overlay (GiB)')
    plt.title('Disk Overlay Usage Over Time')
    plt.legend()
    plt.savefig('disk_overlay.png')
    print("Saved disk_overlay.png")

