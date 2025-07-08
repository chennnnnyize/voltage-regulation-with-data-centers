import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict, deque


def build_sensitivity_matrices(n_buses, edges, r_map, x_map, slack_bus=0):
    """
    Build R and X matrices for linearized DistFlow: v = v0 - 2(Rp + Xq)
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Get list of buses (excluding slack)
    buses = [i for i in range(n_buses) if i != slack_bus]
    n = len(buses)

    # Build path from slack to every bus
    paths = {i: nx.shortest_path(G, slack_bus, i) for i in buses}

    # Initialize R and X matrices
    R = np.zeros((n, n))
    X = np.zeros((n, n))

    bus_to_idx = {bus: idx for idx, bus in enumerate(buses)}

    for i in buses:
        for j in buses:
            pi = paths[i]
            pj = paths[j]
            common_path = set(zip(pi, pi[1:])) & set(zip(pj, pj[1:]))
            rij = sum(r_map[e] for e in common_path)
            xij = sum(x_map[e] for e in common_path)
            R[bus_to_idx[i], bus_to_idx[j]] = rij
            X[bus_to_idx[i], bus_to_idx[j]] = xij

    return R, X, buses

# === Example Usage ===
n_buses = 6
edges = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)]
slack_bus = 0
v0 = 1.0  # squared voltage at slack bus

# Line parameters
r_map = {(i, j): 0.01 for (i, j) in edges}
x_map = {(i, j): 0.02 for (i, j) in edges}

# Build R, X matrices
R, X, buses = build_sensitivity_matrices(n_buses, edges, r_map, x_map, slack_bus)

# === Example power injections ===
# (P, Q) injections at all buses (0 is slack and excluded from calculation)
Pinj_full = np.array([0.0, 3, -0.2, -0.3, 0.0, -0.1])  # example injections
Qinj_full = np.array([0.0, 0.0, -0.1, -0.05, 0.0, -0.03])

Pinj = Pinj_full[buses]
Qinj = Qinj_full[buses]

T = 100
time = np.arange(T)
Pinj_MW_t = np.zeros((T, n_buses))
Pinj_MW_t[:, 1] = -0.1 + 0.002 * np.sin(0.5 * time)
Pinj_MW_t[:, 2] = -0.2 + 0.001 * np.sin(0.2 * time)
Pinj_MW_t[20:, 4] = -2.5 #+ 0.05 * np.cos(0.1 * time)
#Pinj_MW_t[:, 5] = -0.1 #+ 0.03 * np.sin(0.15 * time)
Qinj_MVAR_t = 0.5 * Pinj_MW_t  # constant power factor

inverter_buses = [ 3, 5]
droop_coeff_P = 2.0  # per unit P / volt
droop_coeff_Q = 3.0  # per unit Q / volt



# === Run time series simulation ===
V_history = []
V_history2 = []
Vm = np.ones(n_buses-1)
dV_history=[]
Pinj_previous=np.zeros(n_buses)
Qinj_previous=np.zeros(n_buses)

for t in range(T):
    # Per-unit injections
    Pinj_pu = Pinj_MW_t[t]
    Qinj_pu = Qinj_MVAR_t[t]
    print(Pinj_pu)

    # Initialize voltage estimates for droop
    V_est = np.ones(n_buses)
    V_est[1:] = Vm
    Pinj_orig = np.copy(Pinj_pu[buses])
    Qinj_orig = np.copy(Qinj_pu[buses])
    v_orig = v0 + 2 * (R @ Pinj_orig + X @ Qinj_orig)  # squared voltage
    Vm_orig = np.sqrt(v_orig)  # voltage magnitude
    V_history2.append(Vm_orig.copy())
    dV=np.zeros(n_buses)
    dV[1:]=V_est[1:]-1.0
    # Droop control at inverter buses
    for b in inverter_buses:
        #dV[b] = V_est[b] - 1.0
        Pinj_pu[b] = Pinj_previous[b] -droop_coeff_P * dV[b]
        Qinj_pu[b] = Qinj_previous[b] - droop_coeff_Q * dV[b]
    print("After droop", Pinj_pu)
    print("V_Est", V_est)
    print("DV, ", dV)

    Pinj = Pinj_pu[buses]
    Qinj = Qinj_pu[buses]
    Pinj_previous=np.copy(Pinj_pu)
    Qinj_previous = np.copy(Qinj_pu)



    # === Voltage calculation ===
    v = v0 + 2 * (R @ Pinj + X @ Qinj)  # squared voltage
    Vm = np.sqrt(v)  # voltage magnitude

    # === Print results ===
    print("Bus\tVoltage (p.u.)", Vm)


    V_history.append(Vm.copy())
    dV_history.append(dV.copy())

# === Plot results ===
V_history = np.array(V_history)
V_history2 = np.array(V_history2)
dV_history = np.array(dV_history)
plt.figure(figsize=(10, 6))
'''for i in range(n_buses-1):
    plt.plot(time, V_history[:, i], label=f'Bus {i}')
    plt.plot(time, V_history2[:, i], label=f'Bus_orig {i}')
plt.axhline(1.0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Time (step)')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('Voltage Over Time with Droop Control')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()'''

fig, axs = plt.subplots(n_buses-1, 3, figsize=(10, 2 * n_buses), sharex=True)

for i in range(n_buses-1):
    axs[i,0].plot(time, V_history[:, i], label=f'Bus {i} after VR', color='tab:blue')
    axs[i,0].plot(time, V_history2[:, i], label=f'Bus {i} Uncontrolled', color='tab:red')
    axs[i,1].plot(time, dV_history[:,i+1], label=f'Bus {i} voltage deviation', color='tab:green')
    axs[i,2].plot(time, Pinj_MW_t[:, i+1], label=f'Bus {i} active Power', color='tab:blue')
    axs[i, 2].plot(time, Qinj_MVAR_t[:, i+1], label=f'Bus {i} Reactive Power', color='tab:red')
    #axs[i].axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axs[i,0].set_ylabel('V (p.u.)')
    axs[i,0].set_title(f'Voltage Magnitude at Bus {i}')
    axs[i,0].grid(True)
    axs[i,0].legend()
    axs[i,1].set_ylabel('Delta V (p.u.)')
    axs[i,1].set_title(f'Voltage Deviation at Bus {i}')
    axs[i,1].grid(True)
    axs[i,1].legend()
    axs[i,2].set_ylabel('Active/Reactive Power (p.u.)')
    axs[i,2].set_title(f'P/Q Injection at Bus {i}')
    axs[i,2].grid(True)
    axs[i,2].legend()

axs[-1,0].set_xlabel('Time (step)')
axs[-1,1].set_xlabel('Time (step)')
axs[-1,2].set_xlabel('Time (step)')
plt.tight_layout()
plt.show()


