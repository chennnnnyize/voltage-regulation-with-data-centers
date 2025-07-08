import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from util import read_GPU_Data

def calculate_voltage_sqrt(V0, slack_bus, Pinj_full, Qinj_full):
    # === Build tree structure ===
    children = defaultdict(list)
    parent = {}
    for i, j in edges:
        children[i].append(j)
        parent[j] = i

    # === Step 1: Compute branch flows from injections ===
    P_flow = {}
    Q_flow = {}

    def compute_branch_flows(i):
        p_total = Pinj_full[i]
        q_total = Qinj_full[i]
        for j in children[i]:
            p_child, q_child = compute_branch_flows(j)
            p_total += p_child
            q_total += q_child
            P_flow[(i, j)] = p_child
            Q_flow[(i, j)] = q_child
        return p_total, q_total

    compute_branch_flows(slack_bus)

    # === Step 2: Compute voltages using full DistFlow quadratic ===
    V = np.zeros(n_buses)
    V[slack_bus] = V0  # initialize slack bus voltage

    queue = deque([slack_bus])
    while queue:
        i = queue.popleft()
        for j in children[i]:
            r = r_map[(i, j)]
            x = x_map[(i, j)]
            Pij = P_flow[(i, j)]
            Qij = Q_flow[(i, j)]
            vi = V[i]

            # Coefficients of the quadratic: vj^2 + b*vj + c = 0
            b = 2 * (r * Pij + x * Qij) - vi
            c = (r ** 2 + x ** 2) * (Pij ** 2 + Qij ** 2)
            disc = b ** 2 + 4 * c
            discriminant = np.sqrt(disc)
            if disc < 0:
                raise ValueError(f"Negative discriminant at bus {j}, cannot compute voltage.")
            vj = (-b + discriminant) / 2  # take positive root
            V[j] = vj
            queue.append(j)
    return V

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
v0 = 1.0  # squared voltage at slack bus

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
n_buses = len(mpc['bus'])  # e.g. 123 + added buses
slack_bus = idx_map[114]  # or whichever bus is Slack

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
    r_map[(i, j)] = r*3.0
    x_map[(i, j)] = x*3.0

# Capacitors & regulators
caps = {int(bus): cap for bus, cap in mpc['caps']}
regs = {int(bus): ratio for bus, ratio in mpc['regs']}

# Generator (slack & DER) data
gens = {int(row[0]): row[1:] for row in mpc['gen']}
# e.g. gens[114] gives generator config at slack bus

# Example: specify which buses have inverters
Inverter_buses = [2,  11,  40,  54, 87,  92, 103]  # replace with actual DER bus list
inverter_buses_idx = [idx_map[b] for b in Inverter_buses]
print("inverter buses", inverter_buses_idx)

# Now Pd, Qd, edges, r_map/x_map, inverter_buses_idx are ready for optimization

print(f"{len(bus_ids)} buses loaded, {len(edges)} branches.")
print("Example: Bus 1 load:", Pd[idx_map[1]], Qd[idx_map[1]])


# Loads
Pd = np.zeros(n_buses)
Qd = np.zeros(n_buses)
for b in mpc['bus']:
    idx = idx_map[int(b[0])]
    Pd[idx] = b[2]
    Qd[idx] = b[3]

# Inverter specs – define where you have controllable DERs
inverter_buses = [idx_map[bus] for bus in inverter_buses_idx]
print("Inverter buses", inverter_buses)
P_min = {i: -0.27 for i in inverter_buses}
P_max = {i: +0.27 for i in inverter_buses}
Q_min = {i: -0.3 for i in inverter_buses}
Q_max = {i: +0.3 for i in inverter_buses}

# Build R, X matrices
R, X, buses = build_sensitivity_matrices(n_buses, edges, r_map, x_map, slack_bus)

print("Slack bus", slack_bus)
print("buses", buses)
num_of_DataCenter=8
DC_bus = [13,24, 28, 37,53,77, 85,99]
DC_power=np.array(read_GPU_Data(), dtype=float)


T = 100
time = np.arange(T)
Pinj_MW_t = np.zeros((T, n_buses))
Qinj_MVAR_t = np.zeros((T, n_buses))

for b in mpc['bus']:
    idx = idx_map[int(b[0])]
    Pinj_MW_t[:, idx] = b[2]+np.random.normal(0.0, 0.01)
    Qinj_MVAR_t[:, idx] = b[3]+np.random.normal(0.0, 0.01)

for i in range(num_of_DataCenter):
    time_shift=np.random.randint(100, 220)
    Pinj_MW_t[:, int(DC_bus[i])] = DC_power[time_shift: time_shift+T]/1000.0

#Pinj_MW_t[20:, 3] = -5.5 #+ 0.05 * np.cos(0.1 * time)
#Pinj_MW_t[20:, 4] = -1.5
#Pinj_MW_t[:, 5] = -0.1 #+ 0.03 * np.sin(0.15 * time)
#Qinj_MVAR_t = 0.5 * Pinj_MW_t  # constant power factor

droop_coeff_P = 1.0  # per unit P / volt
droop_coeff_Q = 0.5  # per unit Q / volt
droop_dc_P = 4.0  # per unit P / volt
droop_dc_Q = 2.0  # per unit Q / volt



# === Run time series simulation ===
V_history = []
V_history2 = []
Vm = np.ones(n_buses)
dV_history=[]
Pinj_previous=np.zeros(n_buses)
Qinj_previous=np.zeros(n_buses)
Pinj_previous1=np.zeros(n_buses)
Qinj_previous1=np.zeros(n_buses)

for t in range(T):
    # Per-unit injections
    Pinj_pu = Pinj_MW_t[t]
    Qinj_pu = Qinj_MVAR_t[t]


    #v_orig = v0 + 2 * (R @ Pinj_orig + X @ Qinj_orig)  # squared voltage
    #Vm_orig = np.sqrt(v_orig)  # voltage magnitude
    # Initialize voltage estimates for droop
    V_est = np.ones(n_buses)
    V_est[1:] = Vm[1:]
    dV=np.zeros(n_buses)
    dV[1:]=V_est[1:]-1.0
    # Droop control at inverter buses
    for b in inverter_buses:
        #dV[b] = V_est[b] - 1.0
        Pinj_pu[b] = Pinj_previous1[b] + droop_coeff_P * dV[b]
        Qinj_pu[b] = Qinj_previous1[b] + droop_coeff_Q * dV[b]
    Pinj_previous1 = np.copy(Pinj_pu)
    Qinj_previous1 = np.copy(Qinj_pu)
    Vm_orig = calculate_voltage_sqrt(V0=v0, slack_bus=slack_bus, Pinj_full=Pinj_previous1, Qinj_full=Qinj_previous1)
    V_history2.append(Vm_orig.copy())


    for dc in range(num_of_DataCenter):
        #dV[b] = V_est[b] - 1.0
        Pinj_pu[dc] = Pinj_pu[dc] + droop_dc_P * dV[dc]
        Qinj_pu[dc] = Qinj_pu[dc] + droop_dc_Q * dV[dc]
    #print("After droop", Pinj_pu)
    #print("DV, ", dV)

    Pinj_previous = np.copy(Pinj_pu)
    Qinj_previous = np.copy(Qinj_pu)

    Vm = calculate_voltage_sqrt(V0=v0, slack_bus=slack_bus, Pinj_full=Pinj_previous, Qinj_full=Qinj_previous)

    # === Voltage calculation ===
    #v = v0 + 2 * (R @ Pinj + X @ Qinj)  # squared voltage
    #Vm = np.sqrt(v)  # voltage magnitude

    # === Print results ===
    print("Bus\tVoltage (p.u.)", Vm)


    V_history.append(Vm.copy())
    dV_history.append(dV.copy())

# === Plot results ===


V_history = np.array(V_history)
V_history2 = np.array(V_history2)
dV_history = np.array(dV_history)

for i in range(123):
    plt.plot(V_history[:,i], 'b')
    plt.plot(V_history2[:, i], 'r')
    plt.title(f'Bus {i} voltage')
    plt.show()


fig, axs = plt.subplots(3, 3, figsize=(6.4, 4.8), sharex=True)

axs[0,0].plot(time, V_history[:, 13], label=f'Bus Volt {13}')
axs[0,0].plot(time, V_history2[:, 13], label=f'Bus Volt Original {13}')
axs[0,2].plot(time, Pinj_MW_t[:, 13], label=f'Load {13}')
axs[0,0].axhline(1.0, color='k', linestyle='--', linewidth=1)
axs[0,0].set_xlabel('Time (step)')
axs[0,0].set_ylabel('Voltage Magnitude (p.u.)')
#axs[0,0].title('Voltage Over Time with Droop Control')
axs[0,0].grid(True)
axs[0,0].legend()

axs[0,1].plot(time, V_history[:, 85], label=f'Bus Volt {85}')
axs[0,1].plot(time, V_history2[:, 85], label=f'Bus Volt Original {85}')
axs[0,2].plot(time, Pinj_MW_t[:, 85], label=f'Load {85}')
axs[0,1].axhline(1.0, color='k', linestyle='--', linewidth=1)
axs[0,1].set_xlabel('Time (step)')
axs[0,1].set_ylabel('Voltage Magnitude (p.u.)')
#axs[0,0].title('Voltage Over Time with Droop Control')
axs[0,1].grid(True)
axs[0,1].legend()
axs[0,2].legend()

axs[1,0].plot(time, V_history[:, 4], label=f'Bus Volt {4}')
axs[1,0].plot(time, V_history2[:, 4], label=f'Bus Volt Original {4}')
#axs[1,0].plot(time, Pinj_MW_t[:, 4], label=f'Load {4}')
axs[1,0].axhline(1.0, color='k', linestyle='--', linewidth=1)
axs[1,0].set_xlabel('Time (step)')
axs[1,0].set_ylabel('Voltage Magnitude (p.u.)')
#axs[0,0].title('Voltage Over Time with Droop Control')
axs[1,0].grid(True)
axs[1,0].legend()

axs[1,1].plot(time, V_history[:, 46], label=f'Bus Volt {46}')
axs[1,1].plot(time, V_history2[:, 46], label=f'Bus Volt Original {46}')
#axs[1,1].plot(time, Pinj_MW_t[:, 46], label=f'Load {46}')
axs[1,1].axhline(1.0, color='k', linestyle='--', linewidth=1)
axs[1,1].set_xlabel('Time (step)')
axs[1,1].set_ylabel('Voltage Magnitude (p.u.)')
#axs[0,0].title('Voltage Over Time with Droop Control')
axs[1,1].grid(True)
axs[1,1].legend()

axs[2,0].plot(time, V_history[:, 38], label=f'Bus Volt {38}')
axs[2,0].plot(time, V_history2[:, 38], label=f'Bus Volt Original {38}')
#axs[1,0].plot(time, Pinj_MW_t[:, 4], label=f'Load {4}')
axs[2,0].axhline(1.0, color='k', linestyle='--', linewidth=1)
axs[2,0].set_xlabel('Time (step)')
axs[2,0].set_ylabel('Voltage Magnitude (p.u.)')
#axs[0,0].title('Voltage Over Time with Droop Control')
axs[2,0].grid(True)
axs[2,0].legend()

axs[2,1].plot(time, V_history[:, 86], label=f'Bus Volt {86}')
axs[2,1].plot(time, V_history2[:, 86], label=f'Bus Volt Original {86}')
#axs[1,1].plot(time, Pinj_MW_t[:, 46], label=f'Load {46}')
axs[2,1].axhline(1.0, color='k', linestyle='--', linewidth=1)
axs[2,1].set_xlabel('Time (step)')
axs[2,1].set_ylabel('Voltage Magnitude (p.u.)')
#axs[0,0].title('Voltage Over Time with Droop Control')
axs[2,1].grid(True)
axs[2,1].legend()



plt.tight_layout()
plt.show()

fig, axs = plt.subplots(5, 3, figsize=(6.4, 4.8), sharex=True)

for i in range(5):
    random_num=np.random.randint(1, 123)
    print("Bus ", random_num)
    axs[i,0].plot(time, V_history[:, random_num], label=f'Bus {random_num} after VR', color='tab:blue')
    axs[i,0].plot(time, V_history2[:, random_num], label=f'Bus {random_num} Uncontrolled', color='tab:red')
    axs[i,1].plot(time, dV_history[:,random_num], label=f'Bus {random_num} voltage deviation', color='tab:green')
    axs[i,2].plot(time, Pinj_MW_t[:, random_num], label=f'Bus {random_num} active Power', color='tab:blue')
    axs[i, 2].plot(time, Qinj_MVAR_t[:, random_num], label=f'Bus {random_num} Reactive Power', color='tab:red')
    #axs[i].axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axs[i,0].set_ylabel('V (p.u.)')
    axs[i,0].set_title(f'Voltage Magnitude at Bus {random_num}')
    axs[i,0].grid(True)
    axs[i,0].legend()
    axs[i,1].set_ylabel('Delta V (p.u.)')
    axs[i,1].set_title(f'Voltage Deviation at Bus {random_num}')
    axs[i,1].grid(True)
    #axs[i,1].legend()
    axs[i,2].set_ylabel('Active/Reactive Power (p.u.)')
    axs[i,2].set_title(f'P/Q Injection at Bus {random_num}')
    axs[i,2].grid(True)
    axs[i,2].legend()

axs[-1,0].set_xlabel('Time (step)')
axs[-1,1].set_xlabel('Time (step)')
axs[-1,2].set_xlabel('Time (step)')
plt.tight_layout()
plt.show()


