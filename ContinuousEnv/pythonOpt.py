# example_model.py

import pyomo.environ as pyo

# =============================================================================
# 1. Define Model and Sets
# =============================================================================

model = pyo.ConcreteModel()

# For simplicity, assume a single wind farm with a prediction horizon of N time steps.
N = 6             # prediction horizon (e.g., 6 hours)
T = 1             # sampling time in hours
model.TS = pyo.RangeSet(0, N-1)  # time steps 0, 1, ..., N-1

# Set of jobs arriving within the horizon.
# (In a full implementation, jobs may arrive at different times and be assigned to different wind farms.)
jobs_list = ['job1', 'job2']
model.Jobs = pyo.Set(initialize=jobs_list)

# =============================================================================
# 2. Parameters (dummy values are provided; replace these with your data)
# =============================================================================

# Forecasted wind power generation [kW] for each time step (P_gen in your model)
gen_data = {0: 500, 1: 550, 2: 480, 3: 600, 4: 530, 5: 510}  # example forecast
model.P_gen = pyo.Param(model.TS, initialize=gen_data)

# Economic parameters (all in cents per kWh)
model.c_rev  = pyo.Param(initialize=9.4)    # revenue per kWh injected into grid
model.c_grid = pyo.Param(initialize=22)     # cost per kWh drawn from grid

# HPC power consumption parameters
model.P_idle = pyo.Param(initialize=100)      # idle power per active blade (kW)
model.P_CPU  = pyo.Param(initialize=2)        # additional power per active CPU core (kW)
model.U      = pyo.Param(initialize=8)        # number of CPU cores per blade

# Job-specific parameters.
# For each job, we define:
#    - required core hours (job_core_hours) as U_J in your notation,
#    - arrival time (job_arrival) and deadline (job_deadline).
job_core_hours_data = {'job1': 12, 'job2': 8}    # for example, job1 requires 12 core-hours, job2 8 core-hours.
job_arrival_data    = {'job1': 0, 'job2': 2}       # job1 available at t=0, job2 at t=2.
job_deadline_data   = {'job1': 3, 'job2': 5}       # deadlines (time steps) by which the jobs must be finished.

model.job_core_hours = pyo.Param(model.Jobs, initialize=job_core_hours_data)
model.job_arrival    = pyo.Param(model.Jobs, initialize=job_arrival_data)
model.job_deadline   = pyo.Param(model.Jobs, initialize=job_deadline_data)

# =============================================================================
# 3. Decision Variables
# =============================================================================

# (a) CPU cores scheduled at each time step (the total number of cores used across the HPC windCORE)
model.u = pyo.Var(model.TS, domain=pyo.NonNegativeIntegers)

# (b) Number of active blades at each time step.
# These are integer variables that “cover” the scheduled cores (each blade provides U cores).
model.y = pyo.Var(model.TS, domain=pyo.NonNegativeIntegers)

# (c) Grid power is modeled via its positive and negative parts.
#    P_grid = P_grid_pos - P_grid_neg.
model.P_grid_pos = pyo.Var(model.TS, domain=pyo.NonNegativeReals)  # grid power drawn (if needed)
model.P_grid_neg = pyo.Var(model.TS, domain=pyo.NonNegativeReals)  # wind power sold to grid (if excess)

# (d) Job scheduling variables.
# Let x[j,t] be the (fractional) number of CPU core-hours allocated to job j at time step t.
# (For simplicity we allow these to be continuous; in a detailed model you might need to enforce integrality.)
model.x = pyo.Var(model.Jobs, model.TS, domain=pyo.NonNegativeReals)

# =============================================================================
# 4. Constraints
# =============================================================================

# (4.1) Power Balance Constraint at each time step.
# Your balance equation reads: 
#     P_gen[t] + P_grid[t] = HPC power consumption,
# where HPC power consumption is given by y[t]*P_idle + P_CPU*u[t].
def power_balance_rule(model, t):
    return model.P_gen[t] + (model.P_grid_pos[t] - model.P_grid_neg[t]) == model.y[t] * model.P_idle + model.P_CPU * model.u[t]
model.power_balance = pyo.Constraint(model.TS, rule=power_balance_rule)

# (4.2) Blade Activation Constraint:
# The number of active blades must be enough to supply the scheduled CPU cores.
def blade_activation_rule(model, t):
    return model.u[t] <= model.y[t] * model.U
model.blade_activation = pyo.Constraint(model.TS, rule=blade_activation_rule)

# (4.3) Job Allocation Constraints:
# (a) At each time step, the sum of CPU core-hours allocated to jobs cannot exceed the total scheduled cores.
def job_allocation_rule(model, t):
    # Only jobs that have arrived by time t are eligible.
    return sum(model.x[j, t] for j in model.Jobs if model.job_arrival[j] <= t) <= model.u[t]
model.job_allocation = pyo.Constraint(model.TS, rule=job_allocation_rule)

# (b) For each job, the total allocated core-hours (from its arrival until its deadline) must equal its requirement.
def job_requirement_rule(model, j):
    # Sum over time steps where the job is allowed to run:
    # We assume time is integer and that the job must finish by its deadline.
    return sum(model.x[j, t] for t in model.TS if t >= model.job_arrival[j] and t <= model.job_deadline[j]) == model.job_core_hours[j]
model.job_requirement = pyo.Constraint(model.Jobs, rule=job_requirement_rule)

# =============================================================================
# 5. Objective Function
# =============================================================================

# One possibility is to optimize the energy costs:
# For each time step, we incur a cost for grid consumption and obtain revenue for selling excess wind power.
# Here the net cost (to be minimized) is:
#
#   Sum_t [ c_grid * T * P_grid_pos[t] - c_rev * T * P_grid_neg[t] ]
#
# In a more complete model you would also include terms reflecting QoS (e.g., penalties for missed deadlines) 
# and sustainability metrics.
def objective_rule(model):
    return sum(model.c_grid * T * model.P_grid_pos[t] - model.c_rev * T * model.P_grid_neg[t] for t in model.TS)
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# =============================================================================
# 6. Solve the Model
# =============================================================================

# You can use any MILP solver supported by Pyomo. Here we use GLPK.
solver = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=True)

# =============================================================================
# 7. Display the Results
# =============================================================================

print("\n===== Solution =====")
model.display()
