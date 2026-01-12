import numpy as np

def compute_alpha_schedule(T=500, s=1e-5, clip_min=0.001):
    """
    Computes the recomputed alpha schedule based on the paper's methodology.
    Identical to the previous correct version.
    """
    t_vals = np.arange(0, T + 1)
    
    # Raw schedule f(t)
    f_t = 1 - (t_vals / T)**2
    alpha_raw = (1 - 2*s) * f_t + s
    
    # Recompute with stability clipping
    alpha_recomputed = np.zeros(T + 1)
    current_alpha = 1.0 
    
    for t in range(0, T + 1):
        prev_raw = 1.0 if t == 0 else alpha_raw[t-1]
        curr_raw = alpha_raw[t]
        
        # Calculate ratio squared and clip
        ratio_sq = (curr_raw / prev_raw)**2
        ratio_sq_clipped = max(ratio_sq, clip_min)
        
        # Accumulate
        current_alpha = current_alpha * np.sqrt(ratio_sq_clipped)
        alpha_recomputed[t] = current_alpha
        
    return alpha_recomputed

def get_general_reverse_variance(t, s, alpha_schedule):
    """
    Computes the reverse process variance for the transition z_t -> z_s.
    
    Formula based on Image 2:
    sigma_{t->s} = (sigma_{t|s} * sigma_s) / sigma_t
    
    Args:
        t (int): Current time step.
        s (int): Next time step (target), must be < t.
        alpha_schedule (np.array): Precomputed alpha values.
        
    Returns:
        float: The variance sigma_{t->s}^2.
    """
    if s >= t:
        raise ValueError(f"Target step s ({s}) must be less than current step t ({t})")
    if t >= len(alpha_schedule):
        raise ValueError(f"Step t ({t}) is out of bounds for schedule of length {len(alpha_schedule)}")
        
    # 1. Retrieve Alphas
    alpha_t = alpha_schedule[t]
    alpha_s = alpha_schedule[s]
    
    # 2. Compute basic Sigmas (sigma^2 = 1 - alpha^2)
    # Note: The paper defines alpha_t approx 1 -> 0, so sigma adds noise.
    sigma_t_sq = 1 - alpha_t**2
    sigma_s_sq = 1 - alpha_s**2
    
    # Avoid division by zero if t is very small (though t>s>=0 implies t>=1)
    if sigma_t_sq < 1e-12: 
        return 0.0

    # 3. Compute Transition Terms (Eq 2 details)
    # alpha_{t|s} = alpha_t / alpha_s
    alpha_t_given_s = alpha_t / alpha_s
    
    # sigma_{t|s}^2 = sigma_t^2 - alpha_{t|s}^2 * sigma_s^2
    sigma_t_given_s_sq = sigma_t_sq - (alpha_t_given_s**2 * sigma_s_sq)
    
    # Handle potential negative zero due to float precision
    sigma_t_given_s_sq = max(sigma_t_given_s_sq, 0.0)

    # 4. Compute Posterior Variance (Eq 4 details)
    # Variance = (sigma_{t|s}^2 * sigma_s^2) / sigma_t^2
    variance = (sigma_t_given_s_sq * sigma_s_sq) / sigma_t_sq
    
    return variance

# --- Example Usage with Custom Interval ---

# 1. Generate Schedule
alphas = compute_alpha_schedule(T=500)

# 2. Compute variances for a "Strided" sampling process
# e.g., t = 500 -> 490 -> 480 ... -> 0 (Step size 10)
interval = 10
variances_list = []
steps_log = []

current_t = 500
while current_t > 0:
    target_s = max(current_t - interval, 0) # Ensure s doesn't go below 0
    
    var = get_general_reverse_variance(current_t, target_s, alphas)
    variances_list.append(var)
    steps_log.append(f"{current_t}->{target_s}")
    
    current_t = target_s

# Print first 10 steps of the strided variance
print("Computed Variances for Strided Sampling (first 5 steps):")
for step_desc, val in zip(steps_log[:5], variances_list[:5]):
    print(f"Step {step_desc}: {val:.6e}")