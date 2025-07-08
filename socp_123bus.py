import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# === Setup from your parsed MATPOWER case ===

data_bus = np.loadtxt('Bus_Data.csv', delimiter=',')
data_branch = np.loadtxt('Branch_Data.csv', delimiter=',')
data_generator = np.loadtxt('Generator_Data.csv', delimiter=',')

# Example: paste your parsed data here or load via scipy.io (if .mat)
mpc = {
    'bus': data_bus,
        # ... continue for all buses ..
    'branch': data_branch,
    'caps': np.array([
        # bus, cap_mvar
        [83, 0.600],
        [88, 0.050],
        [90, 0.050],
        [92, 0.050]
    ]),
    'regs': np.array([
        # bus, ratio
        [9, 1.0],
        [26,1.0],
        [67,1.0]
    ]),
    'gen': data_branch,
    'gencost': np.array([
        [2,0,0,3,0,1,0],
        # repeated for others...
    ])
}

# Build index maps
bus_ids = mpc['bus'][:, 0].astype(int)
idx_map = {bus: idx for idx, bus in enumerate(bus_ids)}

# Extract loads
Pd = mpc['bus'][:, 2].copy()
Qd = mpc['bus'][:, 3].copy()

# Extract lines
edges = []
r_map, x_map = {}, {}
for br in mpc['branch']:
    i, j = idx_map[int(br[0])], idx_map[int(br[1])]
    r, x = br[2], br[3]
    edges.append((i, j))
    r_map[(i, j)] = r
    x_map[(i, j)] = x

# Capacitors & regulators
caps = {int(bus): cap for bus, cap in mpc['caps']}
regs = {int(bus): ratio for bus, ratio in mpc['regs']}

# Generator (slack & DER) data
gens = {int(row[0]): row[1:] for row in mpc['gen']}
# e.g. gens[114] gives generator config at slack bus

# Example: specify which buses have inverters
Inverter_buses = [10, 20, 30, 31, 40, 50, 51, 52, 53, 54, 55, 56, 57, 58]  # replace with actual DER bus list
inverter_buses_idx = [idx_map[b] for b in Inverter_buses]
print("inverter buses", inverter_buses_idx)

# Now Pd, Qd, edges, r_map/x_map, inverter_buses_idx are ready for optimization

print(f"{len(bus_ids)} buses loaded, {len(edges)} branches.")
print("Example: Bus 1 load:", Pd[idx_map[1]], Qd[idx_map[1]])

n_buses = len(mpc['bus'])  # e.g. 123 + added buses
bus_ids = [int(b[0]) for b in mpc['bus']]  # bus numbers
idx_map = {bus: idx for idx, bus in enumerate(bus_ids)}
slack = idx_map[114]  # or whichever bus is Slack

# Build edges and line parameters
edges = []
r_map = {}
x_map = {}
for br in mpc['branch']:
    i = idx_map[int(br[0])]
    j = idx_map[int(br[1])]
    r_map[(i, j)] = br[2]
    x_map[(i, j)] = br[3]
    edges.append((i, j))

# Loads
Pd = np.zeros(n_buses)
Qd = np.zeros(n_buses)
for b in mpc['bus']:
    idx = idx_map[int(b[0])]
    Pd[idx] = b[2]
    Qd[idx] = b[3]

print(Pd[47])
Pd[47]=0.4
plt.plot(Pd)
plt.plot(Qd)
plt.show()

# Inverter specs – define where you have controllable DERs
inverter_buses = [idx_map[bus] for bus in inverter_buses_idx]
print("Inverter buses", inverter_buses)
P_min = {i: -0.27 for i in inverter_buses}
P_max = {i: +0.27 for i in inverter_buses}
Q_min = {i: -0.3 for i in inverter_buses}
Q_max = {i: +0.3 for i in inverter_buses}

# === Variables ===
V = cp.Variable(n_buses)  # squared voltages
P = {(i, j): cp.Variable() for (i, j) in edges}
Q = {(i, j): cp.Variable() for (i, j) in edges}
l = {(i, j): cp.Variable() for (i, j) in edges}
Pinj = cp.Variable(n_buses)
Qinj = cp.Variable(n_buses)

# === Objective === minimize voltage magnitude deviation from 1.0
objective = cp.Minimize(cp.sum_squares(V - 1.0))
#objective = cp.Minimize(cp.sum_squares(Qinj))
# === Constraints ===
cons = []

# Slack bus voltage fixed at 1.0²
cons += [V[slack] == 1.0]

# SOCP DistFlow constraints
for (i, j) in edges:
    r, x = r_map[(i, j)], x_map[(i, j)]
    cons += [
        V[j] == V[i] - 2*(r*P[(i, j)] + x*Q[(i, j)]) + (r**2 + x**2)*l[(i, j)],
        cp.SOC(l[(i, j)], cp.hstack([P[(i, j)], Q[(i, j)]]))
    ]

# Nodal power balance
for i in range(n_buses):
    inflow_P = sum(P[(k, j)] for (k, j) in edges if j == i)
    outflow_P = sum(P[(i, j)] for (i, j) in edges if i == i)
    inflow_Q = sum(Q[(k, j)] for (k, j) in edges if j == i)
    outflow_Q = sum(Q[(i, j)] for (i, j) in edges if i == i)

    cons += [
        Pinj[i] + Pd[i] + inflow_P == outflow_P,
        Qinj[i] + Qd[i] + inflow_Q == outflow_Q,
        #V[i] <= 1.1,
        #V[i] >= 0.9
    ]

# Inverter box constraints
for i in inverter_buses:
    cons += [
        P_min[i] <= Pinj[i], Pinj[i] <= P_max[i],
        Q_min[i] <= Qinj[i], Qinj[i] <= Q_max[i]
    ]

# Zero injection at non-inverter nodes
others = set(range(n_buses)) - set(inverter_buses)
for i in others:
    cons += [Pinj[i] == 0, Qinj[i] == 0]

# === Solve ===
prob = cp.Problem(objective, cons)
prob.solve(solver=cp.MOSEK)  # or ECOS/SCS

# === Output ===

print("Optimal voltages (pu):", V.value)
print("Optimal active injections: ", np.round(Pinj.value, 2))
print("Optimal reactive injections: ", np.round(Qinj.value, 2))

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Voltage magnitude plot
axs[0].plot(V.value, color='dodgerblue')
axs[0].axhline(1.0, color='gray', linestyle='--', label='Nominal Voltage')
axs[0].set_ylabel('Voltage (p.u.)')
axs[0].set_title('Voltage Magnitude at Inverter Buses')
axs[0].legend()
axs[0].grid(True)

# Active power injection
print(inverter_buses)
print(Pinj.value[inverter_buses_idx])
#axs[1].bar(inverter_buses, Pinj.value[inverter_buses_idx], color='seagreen')
axs[1].plot(Pinj.value, color='seagreen')
axs[1].set_ylabel('Active Power (p.u.)')
axs[1].set_title('P Injection at Inverter Buses')
axs[1].grid(True)

# Reactive power injection
#axs[2].bar(inverter_buses, Qinj.value[inverter_buses_idx], color='indianred')
axs[2].plot(Qinj.value, color='indianred')
axs[2].set_ylabel('Reactive Power (p.u.)')
axs[2].set_title('Q Injection at Inverter Buses')
axs[2].set_xlabel('Bus Number')
axs[2].grid(True)

plt.tight_layout()
plt.show()





V = np.zeros(n_buses)
V[slack] = 1.0  # squared voltage at slack bus

# Build tree structure
children = defaultdict(list)
for i, j in edges:
    children[i].append(j)

# Forward sweep through the tree
queue = deque([slack])
while queue:
    i = queue.popleft()
    for j in children[i]:
        r, x = r_map[(i, j)], x_map[(i, j)]
        pij, qij = P[(i, j)].value, Q[(i, j)].value
        V[j] = V[i] - 2*(r * pij + x * qij)
        queue.append(j)

Vm = np.sqrt(V)  # voltage magnitude
plt.plot(Vm)
plt.show()


