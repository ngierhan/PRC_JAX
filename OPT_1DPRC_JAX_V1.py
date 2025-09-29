##########
# 1D PRC JAX Optimization
# This code implements a 1D spring-mass-damper (SMD) system using JAX for optimization via parallelization and automatic differentiation.
# Original Simulation Code by: Shan He (shanhe0824@ufl.edu)
# JAX Version and Optimization by: Nyck Gierhan (nvg20@fsu.edu)
##########
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial
import time # for timing the execution
import optax # for the optimizer

jax.config.update("jax_enable_x64", True) # Enable 64-bit precision for JAX to match the original MATLAB code

############ GPU should be default device. Set to CPU if desired. ##########
# jax.config.update('jax_default_device', jax.devices('cpu')[0]) # Set CPU as the default device

NMSErr=0 # 1 = use NMSE for loss, 0 = use R^2 for loss
SingleBeta=1 # 1 = train single beta for all oscillators, 0 = train individual beta for each oscillator
SaveData=0 # 1 = save data to CSV, 0 = do not save data

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
    y_train = y_tar[ind['train']] # target data for training

    # Train the target using Ntrain subsets of oscillators
    dN = param['N']//param['Ntrain'] # # of oscillators per training set
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

        ind_Ntrain = jnp.arange(dN) + nn * dN # oscillators for this training set

        # matrix of training data. Include row of ones to get DC weight
        Xtrain = jnp.vstack([jnp.ones(x[ind_Ntrain][:, ind['train']].shape[1]), x[ind_Ntrain][:, ind['train']]])

        # Determine the weights via ridge regression
        W_mat = Xtrain@Xtrain.T # weight matrix, for determining eigenvalues
        eigenvalues, V = jnp.linalg.eigh(W_mat)
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
    yp_avg = jnp.mean(yp_sep,axis=0) # average over the different training sets

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

##### Cosntraint Function
@jit
def to_constrained_beta(shadow_beta): # Transforms an unconstrained shadow parameter to a non-negative beta
  return jax.nn.softplus(shadow_beta)

############ Main Code ############
key = jax.random.PRNGKey(0) # Initialize a PRNG key
beta_vec = jnp.array([2.0]) # nonlinear spring stiffness (can also be an array of values)

for i in range(jnp.size(beta_vec)):
    # for N in [1, 5, 10, 25, 50, 100, 200, 300, 400, 500]:
    for N in [1, 100, 200]: # Loop over different numbers of oscillators
        beta_option = jnp.array([beta_vec[i], beta_vec[i]]) # mass-normalized nonlinear spirng constant, 0 indicates linear system
        w0 = 1.5 # fundamental frequency
        w1 = 1.5 # linear spring strength (normalized to nat freq)
        Q = 5 # damping factor, (or quality factor)
        zeta = 1/(2*Q) # damping ratio
        newrand = 1 # 1 = new random distr on oscil NL, 0 = load from file

        # Input parameters
        f = 0.5/(2*jnp.pi)*jnp.array([2.11, 3.73, 4.33])   # based on freq from(Nakajima 2015)
        InputType = 'triple' # # of sin() terms in input: 'single' or 'triple', 'triangle, or 'square'.
        T = 1/jnp.max(f) # period of fastest excitation
        Delta_option= jnp.array([0,1]) # strength of input function
        delta_prob = 2 # 1 = 100# excite, 2=50# excited
        Amp = 1 #0.8 # amplitude of forcing input


        # Solution properties
        dt = 0.03 # fixed time step in interpretation. Need to keep fixed for
        numT = 100 # number of periods T to run
        tmax = numT*T # length of run

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
        B22L = jnp.array([0,0,0])*jnp.ones((N,1)) # nonlinear damping

        # linear coupling matrix
        A = jnp.block([[fun_popmat(A11L), fun_popmat(A12L)],
                    [fun_popmat(A21L), fun_popmat(A22L)]
        ])

        Astiff = -fun_popmat(A21L) # dynamics of stiffness & mass: inv(M)*K

# The B matrix is updated inside the training loop and recalculated after each epoch

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
        eigenvalues = jnp.sort(eigenvalues) # sort eigenvalues
        D = jnp.diag(eigenvalues)
        fn = 1/(2*jnp.pi)*jnp.sqrt(jnp.diag(D))

        tspan =jnp.linspace(0,tmax,9674) # time span to run over


        ################################################
        ############# START: ADAM TRAINING #############
        ################################################
############ ADAM training parameters ##########
        learning_rate = 0.01 # Learning rate for ADAM optimizer
        ADAM_num_epochs = 500 # Number of epochs for ADAM optimization
        min_beta = 1
        max_beta = 10
        # num_betas_list = [1,5,10,15,20] 
        num_betas_list = [1] # Number of initial beta values to test

        ############ Verify default device ##########
        tmp = jnp.array([1, 2, 3])
        print(f"Computations using device: {tmp.devices()}")
        time.sleep(3)  # Pause for 3s to allow user to see the device info
        print("--- Starting ADAM optimization for beta ---")
        losses = [] # Store losses for each iteration
        LossResults = [] # Results to save for comparison

        ##### Pre-computation of training data and parameters #####
        # These values do not depend on beta and can be computed once.
        t = jnp.arange(0, tmax + dt, dt)
        F = vmap(fun_force, in_axes=(0, None, None))(t, param, base_key).T
        tempind = jnp.nonzero(ind_Delta, size=1)[0][0]
        I = F[N + tempind, :]
        amp_scale = 0.2 / jnp.max(I)
        # narma_ord = [2,5,10]
        narma_ord = [10] # Order of NARMA to train
        narma_ord_target = narma_ord # Optimize for chosen NARMA function(s)

        # Breakdown of testing vs. training periods
        numT_ignore = int(jnp.round(0.1 * numT))
        numT_train = int(jnp.round(0.5 * numT))
        ind = {}
        ind['ignore'] = t <= numT_ignore * T
        ind['train'] = (t > numT_ignore * T) & (t <= (numT_ignore + numT_train) * T)
        ind['test'] = t > (numT_ignore + numT_train) * T
        ind['tot'] = ind['train'] | ind['test']  # training and test data (combine train and test set)
        
        # Parameters for the training function inside the loss
        param_train = {'lambda_': 0, 'N': N}

        
        ##### Define the loss function #####
        # This function takes beta, simulates the reservoir, and returns the NMSE based on all NARMA function targets
        def create_loss_function(A_const, B11L, B12L, B22L, X0_const, t_const, tspan_const, ind_const, param_const, param_train_const, amp_scale_const, I_const,NMSErr):
            def loss_fn(shadow_beta_dyn):
                beta_dyn = to_constrained_beta(shadow_beta_dyn)
                if beta_dyn.ndim == 0: # If beta is a scalar (0 dimensions)
                    beta_dyn = beta_dyn * jnp.ones(N)
                else: # If beta is already a vector
                    beta_dyn = beta_dyn

                B21L_dyn = jnp.hstack([jnp.zeros((N, 1)), -beta_dyn[:, None], jnp.zeros((N, 1))])
                # Re-build the nonlinear B matrix based on the current beta
                B_dyn = jnp.block([[fun_popmat(B11L), fun_popmat(B12L)],
                                [fun_popmat(B21L_dyn), fun_popmat(B22L)]])

                # Define the system dynamics for the ODE solver
                def fun_Xdot_opt(X, t):
                    force_vector = fun_force(t, param_const, base_key)
                    return A_const @ X + B_dyn @ (X**3) + force_vector

                # Solve the ODE and interpolate
                sol_dyn = odeint(fun_Xdot_opt, X0_const, tspan_const)
                x_ode_dyn = jnp.transpose(sol_dyn)[:N, :]
                x_dyn = vmap(jnp.interp, in_axes=(None, None, 0), out_axes=0)(t_const, tspan_const, x_ode_dyn)

                total_loss = 0.0
                narma_orders_for_loss = narma_ord_target

                # Generate target function for NARMA-2,5,10 and compute loss [sum of loss for each chosen NARMA]
                for n_ord in narma_orders_for_loss:
                    y_tar_dyn = fun_NARMA(amp_scale_const * I_const, n_ord)
                    y_train_dyn = y_tar_dyn[ind_const['train']]
                    Xtrain_dyn = jnp.vstack([jnp.ones(x_dyn[:, ind_const['train']].shape[1]), x_dyn[:, ind_const['train']]])
                    w_dyn = jnp.linalg.solve((Xtrain_dyn @ Xtrain_dyn.T + param_train_const['lambda_'] * jnp.eye(param_train_const['N'] + 1)), Xtrain_dyn @ y_train_dyn.T)
                    yp_dyn = w_dyn[0] + w_dyn[1:].T @ x_dyn

                    if NMSErr == 0:
                        loss = -fun_r2(y_tar_dyn[ind_const['test']], yp_dyn[ind_const['test']])
                    else: # NMSErr == 1
                        loss = fun_nmse(y_tar_dyn[ind_const['test']], yp_dyn[ind_const['test']])
                    total_loss += loss
                
                return total_loss
            return jit(loss_fn)
        
        if NMSErr==0:
            print("Using R2 as loss function")
        else:
            print("Using NMSE as loss function")
        loss_function = create_loss_function(A, B11L, B12L, B22L, X0, t, tspan, ind, param, param_train, amp_scale, I,NMSErr)
        
##### Initialize and run the ADAM optimizer #####
        losses = []
        t_Opt = []
        t_Sim = []
        optimizer = optax.adam(learning_rate)

        if SingleBeta == 1:
            initial_shadow_beta = jnp.log(jnp.expm1(beta[0])) # Optimize a single beta value for all oscillators
        else: # SingleBeta == 0
            initial_shadow_beta = jnp.log(jnp.expm1(beta)) # Optimize individual beta values for each oscillator    

        @jit
        def adam_step(shadow_params, opt_state):
            loss, grads = loss_and_grad_fn(shadow_params)
            updates, new_opt_state = optimizer.update(grads, opt_state, shadow_params)
            new_shadow = optax.apply_updates(shadow_params, updates)
            return new_shadow, new_opt_state, loss
        
        ### Parallelize the training for multiple initial beta values
        def train_for_beta(initial_shadow_scalar):
            if initial_shadow_scalar.ndim == 0: # is scalar
                beta_history_shape = (ADAM_num_epochs,)
            else: # is vector
                beta_history_shape = (ADAM_num_epochs, N)
            opt_state = optimizer.init(initial_shadow_scalar)
            
            # Initialize history arrays for both beta and loss
            beta_history = jnp.zeros(beta_history_shape)
            loss_history = jnp.zeros(ADAM_num_epochs)

            # Define the epoch step function to be used in the loop
            def epoch_step(i, state_tuple):
                shadow, opt_s, b_history, l_history = state_tuple
                # Capture the loss returned by adam_step
                new_shadow, new_opt_s, loss = adam_step(shadow, opt_s)
                # Record both beta and loss values
                b_history = b_history.at[i].set(to_constrained_beta(new_shadow))
                l_history = l_history.at[i].set(loss)
                
                return (new_shadow, new_opt_s, b_history, l_history)

            # Pass both initial history arrays into the loop
            final_shadow, _, final_beta_history, final_loss_history = jax.lax.fori_loop(0, ADAM_num_epochs, epoch_step, (initial_shadow_scalar, opt_state, beta_history, loss_history))
            
            final_loss = final_loss_history[-1]
            final_beta = to_constrained_beta(final_shadow)
            # Return both history arrays
            return final_beta, final_loss, final_beta_history, final_loss_history
        
        for num_betas in num_betas_list:
            if SingleBeta == 1:
                print(f"\n--- Running ADAM optimization of {N} oscillators with {num_betas} initial beta VALUES in parallel for {ADAM_num_epochs} Epochs ---")  
                initial_betas_to_test = jnp.linspace(min_beta,max_beta,num= num_betas)
                print(f"Testing initial beta values in parallel: {initial_betas_to_test}")
            else:
                print(f"\n--- Running ADAM optimization of {N} oscillators with {num_betas} initial beta VECTORS in parallel for {ADAM_num_epochs} Epochs ---")  
                key, subkey = jax.random.split(key)
                initial_betas_to_test = jax.random.uniform(subkey, shape=(num_betas, N), minval=min_beta, maxval=max_beta)
                print(f"Testing initial beta values in parallel: {initial_betas_to_test}")
            loss_and_grad_fn = jax.value_and_grad(loss_function)
            parallel_trainer = jit(vmap(train_for_beta))

            ########## JIT for Accurate Performance Timing ##########
            # print("Jitting parallel Optimization Function...")
            # results = parallel_trainer(initial_betas_to_test)
            # jax.tree_util.tree_map(lambda x: x.block_until_ready(),results) #Ensure function Jitted before timing
            # print("Function Jitted. Starting optimization with performance timing...")
            ########## JIT for Accurate Performance Timing ##########
            initial_shadows_to_test = jnp.log(jnp.expm1(initial_betas_to_test))

            start_time = time.perf_counter()
            all_results = parallel_trainer(initial_shadows_to_test)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), all_results) # Ensure all computations are done
            end_time = time.perf_counter()
            all_trained_betas, all_final_losses, all_beta_histories, all_loss_histories = all_results

            total_time = end_time - start_time
            avg_time_per_epoch = total_time / ADAM_num_epochs
            t_Opt.append([all_beta_histories, all_loss_histories,total_time, avg_time_per_epoch])
            print(f"Total training time for {num_betas} betas: {total_time:.4f} seconds")
            print(f"Average time per epoch for {num_betas} betas: {avg_time_per_epoch:.4f} seconds")

            print("--- ADAM optimization finished. ---")
            
            print(f"Initial Beta: {initial_betas_to_test}")
            print(f"Final Beta: {all_trained_betas}")

##### Save Results for Comparison #####
            if SaveData == 1:
                AllLossesandBetas = [] # Store all losses and betas for comparison
                Indicies = []
                # Loop through each training run
                for i in range(len(initial_betas_to_test)):
                    # Loop through each epoch within that run
                    for epoch in range(ADAM_num_epochs):
                        # Append a row with all the relevant info for that specific epoch
                        AllLossesandBetas.append([
                            i,                                  # Run ID
                            epoch,                              # The current epoch number
                            all_loss_histories[i][epoch],       # The loss at this epoch
                            all_beta_histories[i][epoch],       # The beta value at this epoch
                            total_time,                         # Total time for the entire training run
                            avg_time_per_epoch                  # Average time per epoch  
                        ])
                Indicies.append([ind_beta,
                                ind_Delta
                ])

                import csv
                filename = f'Beta_Opt_BetaShadow_Narma{narma_ord}_{min_beta}_{max_beta}_{ADAM_num_epochs}_R2_V0.95_{N}N_{num_betas}Betas.csv'
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    writer.writerow(['Run_ID', 'Epoch', 'Loss', 'Beta','total_runtime', 'avg_runtime_per_epoch', 'indicies_Beta', 'indicies_Delta'])
                
                    # Write all the data rows
                    writer.writerows([AllLossesandBetas, Indicies])
                print(f"\nData successfully saved to {filename}")
        
    ##############################################
    ############# END: ADAM TRAINING #############
    ##############################################
    

########## Plot Results of Final Beta ##########
    if SingleBeta == 1:
        beta = all_trained_betas* jnp.ones(N) # Use the optimized single beta for all oscillators
        beta = beta.reshape(1, N)
    else:
        beta = all_trained_betas # Individual beta values for each oscillator
    beta = jnp.transpose(beta)

    B21L = jnp.hstack([jnp.zeros((N, 1)), -beta, jnp.zeros((N, 1))])
    # Re-build the nonlinear B matrix based on the current beta
    B = jnp.block([[fun_popmat(B11L), fun_popmat(B12L)],
                    [fun_popmat(B21L), fun_popmat(B22L)]])




##### Run the function once to ensure its Jitted for Performance Testing #####
    # odeint(fun_Xdot, X0, tspan).block_until_ready()
    # print("Function Jitted and ready for testing.")
##### Run the function once to ensure its Jitted for Performance Testing #####

    sol = odeint(fun_Xdot, X0, tspan)
    print("Simulation with Optimized Beta completed.")


    t_ode = tspan # time points
    X = sol # solution at time points

    t_ode = jnp.transpose(t_ode) # change orientation to be same as manual
    X = jnp.transpose(X) # change orientation to be same as manual

    # extract the displacement_i(t) and velocity_i(t)
    x_ode = X[:N,:] # displacements
    xdot_ode= X[N:2*N,:] # dx/dt, velocities
    # Note: 't', 'I', 'ind', 'amp_scale' were already computed before the ADAM block
    x = jnp.zeros((N,len(t)))
    xdot = jnp.zeros((N,len(t)))

    # Interpolation
    interp_vmap = vmap(jnp.interp, in_axes=(None, None, 0), out_axes=0)
    x = interp_vmap(t, t_ode, x_ode)
    xdot = interp_vmap(t, t_ode, xdot_ode)

    # Input and response of select oscillators
    numplot = 4

    # Variables
    Ntrain = 1 # divide N states into this many separate trainings. Total prediction is avg. 1 is using all nodes for training
    lambda_ = 0   # order of ridge regression. 0 yields linear regression
    narma_ord = [2,5,10]


    ## Training (NARMA)
    # taken from Nakajima (2015)
    # The 'ind' dictionary and 'amp_scale' are already defined.

    param['N'] = N
    param['Ntrain'] = Ntrain
    param['lambda_'] = lambda_
    yp_tar = []
    yp_avg_list = []
    yp_sep_list = []
    error_list = []
    weight_list = []
    print(f"Training with {Ntrain} training sets and NARMA orders: {narma_ord} using OPTIMIZED beta")

    for j, n_ord in enumerate(narma_ord):
        curr_y_tar = fun_NARMA(amp_scale*I,narma_ord[j]) # Calculate target function
        param['n'] = narma_ord[j]
        yp_avg_i,yp_sep_i,error_i,weight_i, D = fun_Train(t,curr_y_tar,x,ind,param) # Training

        yp_tar.append(curr_y_tar) # store target function
        yp_avg_list.append(yp_avg_i)
        yp_sep_list.append(yp_sep_i)
        error_list.append(error_i)
        weight_list.append(weight_i)


########## Plotting ##########
## Plot ADAM Loss
losses = all_loss_histories[:,None][0][0]
print(losses)
epochs = jnp.arange(len(losses))+1
print(epochs)

plt.figure(figsize=(10, 6))
if NMSErr==0:
    plt.plot(epochs, losses, label='Loss (R2)', color='blue')
    plt.ylabel('Loss (R2)')
else:
    plt.plot(epochs, losses, label='Loss (NMSE)', color='blue')
    plt.ylabel('Loss (NMSE)')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()
plt.show(block=False) # Show the plot without blocking execution

## Plot NARMA function fits
fig, axs = plt.subplot_mosaic([['narma2'], ['narma5'], ['narma10']], figsize=(12, 15), facecolor='w', sharex=True)
fig.suptitle('Performance of Single Optimized Beta on Individual NARMA Tasks', fontsize=16)

narma_orders_for_readout = narma_ord
param['Ntrain'] = 1
param['lambda_'] = 0

# Store results for each NARMA order in lists
all_yp_tar = []
all_yp_avg = []
all_errors = []

print(f"Training readout for NARMA orders: {narma_orders_for_readout} using the SINGLE OPTIMIZED beta")
for n_ord in narma_orders_for_readout:
    curr_y_tar = fun_NARMA(amp_scale*I, n_ord)
    param['n'] = n_ord
    yp_avg_i, _, error_i, _, _ = fun_Train(t, curr_y_tar, x, ind, param)
    all_yp_tar.append(curr_y_tar)
    all_yp_avg.append(yp_avg_i)
    all_errors.append(error_i)

for i, n_ord in enumerate(narma_orders_for_readout):
    ax = axs[f'narma{n_ord}']
    yp_tar_i = all_yp_tar[i]
    yp_avg_i = all_yp_avg[i]
    error_i = all_errors[i]

    line1, = ax.plot(t[ind['train'] | ind['test']], yp_tar_i[ind['train'] | ind['test']], linewidth=2, label='Target')
    line2, = ax.plot(t[ind['train'] | ind['test']], yp_avg_i[ind['train'] | ind['test']], linewidth=2, label='Predicted', zorder=10)
    t_break = t[jnp.where(ind['test'])[0][0]]
    ax.axvline(x=t_break, color='k', linestyle='-')
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.grid(True)
    ax.legend()
    title_line1 = f'NARMA-{n_ord} Prediction'
    title_line2 = (f'NMSE_train={error_i["nmse_train_avg"]:.1e}, '
                    f'NMSE_test={error_i["nmse_test_avg"]:.1e}, '
                    f'R2_train={error_i["r2_train_avg"]:.2f}, '
                    f'R2_test={error_i["r2_test_avg"]:.2f}')
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()