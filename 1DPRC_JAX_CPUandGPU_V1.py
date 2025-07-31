##########
# 1D PRC JAX Implementation
# This code implements a 1D spring-mass-damper (SMD) system using JAX for parallelization and automatic differentiation.
# Original Code by: Shan He (shanhe0824@ufl.edu)
# JAX Version by: Nyck Gierhan (nvg20@fsu.edu)
##########
### My Hardware:
### CPU: 13th Gen Intel Core i7-13700KF
### GPU: NVIDIA GeForce RTX 4070 Ti

### My Results:
### 378 seconds for 500 sequential simulations on CPU
### 186 seconds for 500 parallel simulations on GPU


import jax
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial
import time # For timing the execution

jax.config.update("jax_enable_x64", True) # Enable 64-bit precision for JAX to match the original MATLAB code

Parallelize = True  # Set to True to enable parallel simulations

############ GPU should be default device. Set to CPU if desired. ##########
jax.config.update('jax_default_device', jax.devices('cpu')[0]) # Set CPU as the default device

############ Uncomment to Verify default device ##########
# tmp = jnp.array([1, 2, 3])
# print(f"Computations using device: {tmp.devices()}")
# breakpoint() # Initiate Python debugger to inspect the device


############ Functions ############
@jit # JIT compile for performance
def fun_popmat(v):
    # assumes v is a [Nx3] matrix
    # populate v onto an [NxN] matrix with v terms below, on, and above the
    # diagonal
    N = v.shape[0]
    temp1 = jnp.diag(v[1:N, 0], k=-1) # below diagonal (coupling to left oscillator)
    temp2 = jnp.diag(v[:, 1]) # diagonal (self-coupling)
    temp3 = jnp.diag(v[:N-1, 2], k=1) # above diagonal (coupling to right oscillator)
    Mat = temp1 + temp2 + temp3 # populated matrix
    return Mat

## Forcing (Input) function
@jit # JIT compile for performance
def jax_triangle_wave(x):
  return (2 / jnp.pi) * jnp.arcsin(jnp.sin(x))

@jit # JIT compile for performance
def jax_square_wave(x):
  val = jnp.sin(x)
  return jnp.where(val == 0, 1.0, jnp.sign(val))

def fun_force(t, param, base_key):
    
    def single_force(operand):
        t_, _ = operand
        return param['Amp'] * param['Delta'] * jnp.sin(2 * jnp.pi * param['f'][0] * t_)
    
    def triple_force(operand):
        t_, _ = operand
        return param['Amp'] * param['Delta'] * jnp.sin(2 * jnp.pi * param['f'][0] * t_) * jnp.sin(2 * jnp.pi * param['f'][1] * t_) * jnp.sin(2 * jnp.pi * param['f'][2] * t_)
        
    def triangle_force(operand):
        t_, _ = operand
        arg = 2 * jnp.pi * param['f'][0] * t_
        return param['Amp'] * param['Delta'] * jax_triangle_wave(arg)

    def square_force(operand):
        t_, _ = operand
        arg = 2 * jnp.pi * param['f'][0] * t_
        return param['Amp'] * param['Delta'] * jax_square_wave(arg)
        
    def noisy_triple_force(operand):
        t_, key = operand 
          
        f2_signal = jnp.sin(2 * jnp.pi * param['f'][0] * t_) * jnp.sin(2 * jnp.pi * param['f'][1] * t_) * jnp.sin(2 * jnp.pi * param['f'][2] * t_)
        std_dev = 0.01 * jnp.max(f2_signal)
        noise = std_dev * jax.random.normal(key, shape=t_.shape, dtype=t_.dtype)

        return param['Amp'] * param['Delta'] * (f2_signal + noise)

    # List of functions and index for lax.switch
    branch_fns = [single_force, triple_force, triangle_force, square_force, noisy_triple_force]
    branch_names = ['single', 'triple', 'triangle', 'square', 'noisy_triple']
    branch_index = branch_names.index(param['InputType'])
    
    operand = (t, base_key)
    
    # Use lax.switch to execute the correct branch efficiently
    F2 = jax.lax.switch(branch_index, branch_fns, operand)
    
    F1 = jnp.zeros_like(F2)
    F = jnp.concatenate([F1, F2], axis=0)
    return F

## Narma function
@partial(jit, static_argnums=(1,)) # JIT compile for performance
def fun_NARMA(I, n):

    # alpha, beta, gamma, delta values
    varnar = jax.lax.cond(
        n == 2,
        lambda: jnp.array([0.4, 0.4, 0.6, 0.1]),
        lambda: jnp.array([0.3, 0.05, 1.5, 0.1])
    )

    def _narma_step(carry, i_curr):
        y_hist, i_hist = carry
        y_prev = y_hist[0]

        def n2_update():
            # Recurrence: y_t = v0*y_{t-1} + v1*y_{t-1}*y_{t-2} + v2*I_{t-1}**3 + v3
            y_prev_prev = y_hist[1]
            i_prev = i_hist[0]
            return varnar[0]*y_prev + varnar[1]*y_prev*y_prev_prev + varnar[2]*i_prev**3 + varnar[3]

        def nN_update():
            # Recurrence: y_t = v0*y_{t-1} + v1*y_{t-1}*sum(y_{t-n..t-1}) + v2*I_{t-1}*I_{t-n} + v3
            i_prev = i_hist[0]
            i_prev_n = i_hist[-1]
            return varnar[0]*y_prev + varnar[1]*y_prev*jnp.sum(y_hist) + varnar[2]*i_prev*i_prev_n + varnar[3]
        
        y_next = jax.lax.cond(n == 2, n2_update, nN_update)

        # Update histories by shifting in the new values
        new_y_hist = jnp.roll(y_hist, 1).at[0].set(y_next)
        new_i_hist = jnp.roll(i_hist, 1).at[0].set(i_curr)
        return (new_y_hist, new_i_hist), y_next

    # Initialize history buffers and run the scan
    initial_carry = (jnp.zeros_like(I, shape=n), jnp.zeros_like(I, shape=n))
    _, y_out = jax.lax.scan(_narma_step, initial_carry, I)
    
    # Prepend a zero to match the original function's output, where y[0] is always 0
    return jnp.concatenate([jnp.array([0.0]), y_out[:-1]])


## Error functions 
@jit # JIT compile for performance
def fun_nmse(y_tar,y_p):
    nmse = jnp.sum((y_tar-y_p)**2)/jnp.sum(y_tar**2)
    return nmse

@jit # JIT compile for performance
def fun_r2(y_tar,y_p):
    ymean = jnp.mean(y_tar) 
    r2 = 1-jnp.sum((y_tar-y_p)**2)/jnp.sum((y_tar-ymean)**2) 
    return r2



## Training Function      
def fun_Train(t,y_tar,x,ind,param):
    y_train = y_tar[ind['train']]  # target data for training 
    
    # # # Train the target using Ntrain subsets of oscillators 
    dN = param['N']//param['Ntrain']  # # of oscillators per training set 
    yp_sep_list = []
    weight_list = []
    # Create a dictionary to hold all errors
    error = {
        'narma_ord': param['n'],
        'nmse_train_sep': [],
        'nmse_test_sep': [],
        'r2_train_sep': [],
        'r2_test_sep': []
    }
    for nn in range(param['Ntrain']):
        
        ind_Ntrain = jnp.arange(dN) + nn * dN  # oscillators for this training set
    
        # matrix of training data. Include row of ones to get DC weight 
        Xtrain = jnp.vstack([jnp.ones(x[ind_Ntrain][:, ind['train']].shape[1]), x[ind_Ntrain][:, ind['train']]]) 
        
        # Determine the weights via ridge regression
        W_mat = Xtrain@Xtrain.T  #weight matrix, for determining eigenvalues
        eigenvalues, V = jnp.linalg.eig(W_mat)
        D = jnp.diag(eigenvalues)
        w = jnp.linalg.solve((Xtrain@Xtrain.T+param['lambda_']*jnp.eye(dN+1)),Xtrain@y_train.T)
        woff = w[0]
        wlin = w[1:] # pull out dc offset and linear weights 

        # Use the 1D array 'ind_Ntrain' directly to select rows from x.
        yp_sep_i = woff + wlin.T @ x[ind_Ntrain, :]
        yp_sep_list.append(yp_sep_i)
        
        error['nmse_train_sep'].append(fun_nmse(y_tar[ind['train']],yp_sep_list[nn][ind['train']]))
        error['nmse_test_sep'].append(fun_nmse(y_tar[ind['test']],yp_sep_list[nn][ind['test']]))
        error['r2_train_sep'].append(fun_r2(y_tar[ind['train']],yp_sep_list[nn][ind['train']]))
        error['r2_test_sep'].append(fun_r2(y_tar[ind['test']],yp_sep_list[nn][ind['test']]))
        
    yp_sep = jnp.vstack(yp_sep_list)
    yp_avg = jnp.mean(yp_sep,axis=0)  # average over the different training sets 

    error['nmse_train_avg'] = fun_nmse(y_tar[ind['train']],yp_avg[ind['train']])
    error['nmse_test_avg'] = fun_nmse(y_tar[ind['test']],yp_avg[ind['test']]) 
    error['r2_train_avg'] = fun_r2(y_tar[ind['train']],yp_avg[ind['train']])
    error['r2_test_avg'] = fun_r2(y_tar[ind['test']],yp_avg[ind['test']])

    # save one set of weights. Use to see if I'm applifying small deviations in
    # states too much (Lukosevicius 4.2) 
    weight = w.T
    return yp_avg,yp_sep,error,weight,D

@jit # JIT compile for performance
def fun_Xdot(X,t):
    return A @ X + B @ (X**3) + fun_force(t, param, base_key)

############ Main Code ############
key = jax.random.PRNGKey(0) # Initialize a PRNG key
beta_vec = jnp.array([2.0])  # nonlinear spring stiffness (can also be an array of values)

for i in range(jnp.size(beta_vec)):
    N = 200  # number of masses/nodes. 
    # N = 10  # number of masses/nodes. 
    beta_option = jnp.array([beta_vec[i], beta_vec[i]]) # mass-normalized nonlinear spirng constant, 0 indicates linear system
    w0 = 1.5  # fundamental frequency
    w1 = 1.5  # linear spring strength (normalized to nat freq)
    Q = 5  # damping factor, (or quality factor)
    zeta = 1/(2*Q)  # damping ratio 
    # beta_option = [0.05,1]  # Two levels of nonlinearity (weak, strong) 
    newrand = 1  # 1 = new random distr on oscil NL, 0 = load from file

    # Input parameters
    f = 0.5/(2*jnp.pi)*jnp.array([2.11, 3.73, 4.33])   # based on freq from(Nakajima 2015)
    InputType = 'triple'  # # of sin() terms in input: 'single' or 'triple', 'triangle, or 'square'.
    T = 1/jnp.max(f)  # period of fastest excitation
    Delta_option= jnp.array([0,1])  # strength of input function
    delta_prob = 2  # 1 = 100# excite, 2=50# excited
    Amp = 1  #0.8  # amplitude of forcing input


    # Solution properties
    dt = 0.03  # fixed time step in interpretation. Need to keep fixed for 
    numT = 100  # number of periods T to run 
    tmax = numT*T  # length of run  

    ## Random Distribution on beta and Delta 
    # Determine the random distributions of oscillators for beta and Delta. 
    # Load from file or generate new distribution 
    if newrand == 1:
        key, subkey1, subkey2, base_key = jax.random.split(key, 4)
        ind_beta = jax.random.randint(subkey1, (N,), 0, 4) == 0 # indices with strong nonlinearity (25# chance)
        ind_Delta = jax.random.randint(subkey2, (N,), 0, delta_prob) == 0 # indices with an input(50# chance)



    ### Nonlinearities, beta 
    beta = beta_option[0]*jnp.ones(N)
    beta = beta.at[ind_beta].set(beta_option[1])

    Delta= Delta_option[0]*jnp.ones(N)
    Delta = Delta.at[ind_Delta].set(Delta_option[1])


    ## Define reservoir dynamics
    # These are the 2nd order differential equations for N coupled harmonic
    # oscillators. Rewritten in state-space form

    # Variable Definition
    # A11L = zeros
    # A12L = diagonal, convertion from 2nd order to state-space
    # A21L = stiffness coupling for each ODE
    # A22L = damping coupling for each ODE

    # linear parameters for each ODE
    A11L = jnp.array([0,0,0])*jnp.ones((N,1))
    A12L = jnp.array([0,1,0])*jnp.ones((N,1))
    A21L = jnp.array([w1**2, -(w0**2+2*w1**2), w1**2])*jnp.ones((N,1)) # stiffness coupling for each oscillator
    A22L = jnp.array([0, -w0/Q, 0])*jnp.ones((N,1)) # damping coupling for each oscillator

    # nonlinear parameters for each ODE
    B11L = jnp.array([0,0,0])*jnp.ones((N,1)) 
    B12L = jnp.array([0,0,0])*jnp.ones((N,1)) 
    B21L = jnp.hstack([jnp.zeros((N,1)),-beta[:, None],jnp.zeros((N,1))])  # nonlinear stiffness for each oscillator
    B22L = jnp.array([0,0,0])*jnp.ones((N,1))  # nonlinear damping

    # linear coupling matrix
    A = jnp.block([[fun_popmat(A11L), fun_popmat(A12L)], 
                [fun_popmat(A21L), fun_popmat(A22L)]
    ])
    
    Astiff = -fun_popmat(A21L)  # dynamics of stiffness & mass: inv(M)*K 
    
    # nonlinear coupling matrix
    # Run each function call separately
    B11 = fun_popmat(B11L)
    B12 = fun_popmat(B12L)
    B21 = fun_popmat(B21L)
    B22 = fun_popmat(B22L)

    B = jnp.block([[fun_popmat(B11L), fun_popmat(B12L)],
                [fun_popmat(B21L), fun_popmat(B22L)]
    ]) 
    

    # State Space Representation of the System (called forcing fcn for ode
    # solvers)
    param = {} # Initialize parameter dictionary
    param['Amp'] = Amp  
    param['Delta'] = Delta
    param['f'] = f  
    param['N'] = N 
    param['InputType'] = InputType  
    

    # initial conditions. Need some to start off the system. 
    # enable ode15s() to find the appropriate time steps 
    X0 = 1/20*Amp*jnp.ones(2*N)


    ## Eigenvalues of the linear reservoir 
    eigenvalues, V = jnp.linalg.eigh(Astiff)
    eigenvalues = jnp.sort(eigenvalues)  # sort eigenvalues
    D = jnp.diag(eigenvalues)
    fn = 1/(2*jnp.pi)*jnp.sqrt(jnp.diag(D))

    tspan =jnp.linspace(0,tmax,9674)  # time span to run over

########## SEQUENTIAL SIMULATIONS ##########
    if not Parallelize:
        print(f"Running Sequential Simulations Using Device {tspan.devices()}...")
        # Run the function once to ensure its Jitted
        odeint(fun_Xdot, X0, tspan).block_until_ready()
        print("Function Jitted and ready for performance testing.")

        num_sims = [1, 10, 100, 500]
        PerfResults = []
        for count in num_sims:
            print(f"--- Running for {count} simulation(s) ---")
            start_time = time.perf_counter() # Start the timer

            # sol = odeint(fun_Xdot, X0, tspan) # single simulation
            for _ in range(count): # num_sims simulations
                sol = odeint(fun_Xdot, X0, tspan)
            sol.block_until_ready() # Ensure all computations are complete before measuring time
            end_time = time.perf_counter() # End the timer

            print(f"ODE integration took {end_time - start_time:.4f} seconds") # Print the time taken for the simulations
            PerfResults.append([count, end_time - start_time]) # Store performance results
        print("Sequential simulations completed.")
########## SEQUENTIAL SIMULATIONS ##########



########## PARALLEL SIMULATIONS ##########
    if Parallelize:
        print("Running Parallel Simulations...")

        # Ensure the GPU is available and set the device
        if jax.devices("cuda"):
            print("Using GPU for parallel simulations.")
            gpu_device = jax.devices("cuda")[0]
            cpu_device = jax.devices("cpu")[0]  # Store extra data on CPU if needed
        else:
            print("No GPU available, using CPU for parallel simulations.")

    ## Modify Solver for Parallelization
        def create_dynamics_func(A, B, param, base_key):
            def fun_Xdot_closure(X, t):
                force_vector = fun_force(t, param, base_key)
                return A @ X + B @ (X**3) + force_vector 
            return jit(fun_Xdot_closure)
        
        fun_Xdot = create_dynamics_func(A, B, param, base_key)

        def single_solver(x0):
            return odeint(fun_Xdot, x0, tspan)

        parallel_solver = vmap(single_solver)

    ## Warmup to JIT compile the function
        warmup_batch = jnp.zeros((2, 2 * N)) 
        print("Jitting function...")
        parallel_solver(warmup_batch).block_until_ready() # Solve a dummy batch to JIT compile the function
        print("Function Jitted and ready for performance testing.")

    ## Parallelize the simulations
        num_sims = [1, 10, 100, 500]
        PerfResults = []

        # Modify "max_count" and "batch_size" to control batches if GPU memory is an issue
        max_count = 500  # Maximum number of simulations to run in parallel simultaneously
        batch_size = 500  # Size of each batch for parallel processing

        for count in num_sims:
            key, subkey = jax.random.split(key)
            batch_of_X0s = (1/20 * Amp)*jax.random.normal(subkey, shape=(count, 2*N)) # Create all initial conditions for parallel processing

            print(f"--- Running for {count} simulation(s) ---")

            start_time = time.perf_counter() # Start the timer
            if count <= max_count:
                all_solutions_parallel = parallel_solver(batch_of_X0s).block_until_ready()
                solutions_cpu = jax.device_put(all_solutions_parallel, cpu_device) # Move the results to CPU to clear up GPU RAM
            else:
                batched_solutions = []
                for i in range(0, count, batch_size):
                    batch_chunk = batch_of_X0s [i:i + batch_size]
                    solution_chunk = parallel_solver(batch_chunk).block_until_ready()
                    batched_solutions_cpu = jax.device_put(solution_chunk,  cpu_device) 
                all_solutions_parallel = jnp.concatenate(batched_solutions_cpu, axis=0)
            end_time = time.perf_counter() # End the timer

            PerfResults.append([count, end_time - start_time]) # Store performance results
            print(f"ODE integration took {end_time - start_time:.4f} seconds") # Print the time taken for the simulations
        print("Parallel Simulations Completed.")
        sol = all_solutions_parallel[0]  # Use the first solution as the final result for training and plotting
########### PARALLEL SIMULATIONS ##########

########## ORGANIZE AND TRAIN DATA ##########
    t_ode = tspan  # time points
    X = sol  # solution at time points
    
    t_ode = jnp.transpose(t_ode)  # change orientation to be same as manual
    X = jnp.transpose(X)  # change orientation to be same as manual

    # extract the displacement_i(t) and velocity_i(t)
    x_ode = X[:N,:]  # displacements
    xdot_ode= X[N:2*N,:]  # dx/dt, velocities
    t = jnp.arange(0, tmax + dt, dt)
    numt = len(t)
    x = jnp.zeros((N,len(t)))
    xdot = jnp.zeros((N,len(t)))
    
    # Interpolation
    interp_vmap = vmap(jnp.interp, in_axes=(None, None, 0), out_axes=0)
    x = interp_vmap(t, t_ode, x_ode)
    xdot = interp_vmap(t, t_ode, xdot_ode)
    

    # Generate forcing that was used 
    F = vmap(fun_force,in_axes=(0,None,None))(t,param, base_key).T 
    tempind = jnp.nonzero(ind_Delta, size=1)[0][0]  # index of nonzero input 
    I = F[N+tempind,:]  # input signal 

    # Input and response of select oscillators 
    numplot = 4; 

    # Breakdown of testing vs. training periods (and ignore) 
    numT_ignore = int(jnp.round(0.1*numT)) 
    numT_train = int(jnp.round(0.5*numT))
    numT_test = numT-numT_ignore-numT_train 

    # Variables 
    Ntrain = 1  # divide N states into this many separate trainings. Total prediction is avg. 1 is using all nodes for training
    lambda_ = 0   # order of ridge regression. 0 yields linear regression 
    narma_ord = [2, 5, 10] # order of narma 


    ## Training (NARMA) 
    # taken from Nakajima (2015)

    # scale factor to apply to input before feeding into Narma. Necessary so
    # that it doesn't blow up 
    amp_scale = 0.2/jnp.max(I)
        
    # indices of training vs. test data (and data to ignore)
    ind = {}  # initialize dictionary for indices
    #create boolean function for each portion
    ind['ignore'] = t<=numT_ignore*T  # indices at start to ignore 
    ind['train'] = (t>numT_ignore*T) & (t<=(numT_ignore+numT_train)*T)  # indices of training data 
    ind['test'] = t>(numT_ignore+numT_train)*T  # indices of test data 
    ind['tot'] = ind['train'] | ind['test']  # training and test data (combine train and test set)

    param['N'] = N 
    param['Ntrain'] = Ntrain
    param['lambda_'] = lambda_ 
    yp_tar = []
    yp_avg_list = []
    yp_sep_list = []
    error_list = []
    weight_list = []
    print(f"Training with {Ntrain} training sets and NARMA orders: {narma_ord}")
    
    for j, n_ord in enumerate(narma_ord):
        curr_y_tar = fun_NARMA(amp_scale*I,narma_ord[j]) # Calculate target function 
        param['n'] = narma_ord[j]
        yp_avg_i,yp_sep_i,error_i,weight_i, D = fun_Train(t,curr_y_tar,x,ind,param) # Training 

        yp_tar.append(curr_y_tar)  # store target function
        yp_avg_list.append(yp_avg_i)
        yp_sep_list.append(yp_sep_i)
        error_list.append(error_i)
        weight_list.append(weight_i)


########## Plotting ##########
## Plot 1 PRC prediciton performance results 
t_tot_ind = jnp.where(ind['tot'])[0]  # indices of total data (train + test)
x_start = t[t_tot_ind[0]]  # start time of total data
x_end = t[t_tot_ind[-1]]  # end time of total data
t_break = t[jnp.where(ind['test'])[0][0]]  # time of break between train and test data

fig, axs = plt.subplot_mosaic(
    [['input', 'states'],
     ['output1', 'output1'],
     ['output2', 'output2'],
     ['output3', 'output3']],
    figsize=(12,10), facecolor='w', sharex=True)

ax = axs['input']
ax.plot(t[ind['tot']], I[ind['tot']])
ax.axvline(x=t_break, color='k', linestyle='-')
ax.set_xlabel('t')
ax.set_ylabel('I(t)')
ax.set_title('Input, I(t)')
ax.grid(True)
ax.set_xlim([x_start, x_end])

ax = axs['states']
numplot = 8  # number of oscillators to plot
osc_plot = jax.random.randint(key, (numplot,), minval=0, maxval=N)
for osc_index in osc_plot:
    ax.plot(t[ind['tot']], x[osc_index, ind['tot']], label=f'n={osc_index}')
ax.axvline(x=t_break, color='k', linestyle='-')
ax.set_xlabel('t')
ax.set_ylabel('x(t)')
ax.set_title('States, x(t)')
ax.grid(True)
ax.set_xlim([x_start, x_end])

for i, n_ord in enumerate(narma_ord):
    ax = axs[f'output{i+1}']

    line1, = ax.plot(t[ind['tot']], yp_tar[i][ind['tot']], linewidth=2, label='Target')
    line2, = ax.plot(t[ind['tot']], yp_avg_list[i][ind['tot']], linewidth=2, label='Predicted', zorder=10)

    for nn in range(Ntrain):
        ax.plot(t[ind['tot']], yp_sep_list[i][nn, ind['tot']], '--', linewidth=0.5, color='gray')

    ax.axvline(x=t_break, color='k', linestyle='-')
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.grid(True)
    ax.set_xlim([x_start, x_end])
    ax.legend(handles=[line1, line2])

    title_line1 = f'Output, Narma {n_ord}'
    title_line2 = (r'$NMSE_{train}$'f'={error_list[i]["nmse_train_avg"]:.1e}, '
                   r'$NMSE_{test}$'f'={error_list[i]["nmse_test_avg"]:.1e}, '
                   r'$R^2_{train}$'f'={error_list[i]["r2_train_avg"]:.2f}, '
                   r'$R^2_{test}$'f'={error_list[i]["r2_test_avg"]:.2f}')
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=10)   
    
plt.show() 


