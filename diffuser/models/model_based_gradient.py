import torch
import diffuser.utils as utils

def model_based_score(x_t, normalizer, wall_loc, cond, alphas_cumprod_t, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod):
    """
    calculate score from model-based diffusion

    param x_t: batch of path x at corresponding step t, [batch_size, horizon=48, dims=2]
    param normalizer: normalizer of diffuser
    param wall_loc: batch of wall location of corresponding x, [batch_size, num_wall * dims]
    param cond: batch cond of corresponding x, in this case: start and end
    
    return score
    """
    device = wall_loc.device
    batch_size, horizon, dimension = x_t.shape
    wall_loc = wall_loc.reshape(batch_size, -1, dimension) # [batch_size, num_wall, dims]


    """
    generate noise to sample x0(t)
    """
    Nsample = 20000
    noise = torch.randn(batch_size, Nsample, horizon, dimension).to(device)

    x_t_expanded = x_t.unsqueeze(1).expand(-1, Nsample, -1, -1).to(device) # (batch_size, Nsample, horizon, dimension)
    sigma = sqrt_one_minus_alphas_cumprod.unsqueeze(-1) / sqrt_alphas_cumprod_t.unsqueeze(-1)
    x_0_samples = noise * sigma + x_t_expanded / sqrt_alphas_cumprod_t.unsqueeze(-1)

    # start and end should be the same with cond
    x_0_samples[:,:,0,:] = x_t_expanded[:,:,0,:] 
    x_0_samples[:,:,-1,:] = x_t_expanded[:,:,-1,:] 
    
    x_0_samples = torch.clamp(x_0_samples, -1.0, 1.0)  

    x_0_samples_unnormalized = utils.to_np(x_0_samples)
    x_0_samples_unnormalized = normalizer.unnormalize(x_0_samples_unnormalized, 'observations')
    x_0_samples_unnormalized = torch.from_numpy(x_0_samples_unnormalized).to(device)

    """
    calculate log p for each x_0 sample
    """
    temp_sample = 0.1 
    # calculate reward
    rewards = compute_reward(x_0_samples_unnormalized, wall_loc)  #(batch_size, Nsample)

    rewards_std = rewards.std(dim=1, keepdim=True)
    rewards_std = torch.where(rewards_std < 1e-4, torch.tensor(1.0), rewards_std)  
    rewards_mean = rewards.mean(dim=1, keepdim=True)
    logp0 = (rewards - rewards_mean) / (rewards_std * temp_sample)  #(batch_size, Nsample)

    # print("Y0s shape:", x_0_samples.shape)  
    # print("logp0 shape:", logp0.shape) 

    # use softmax to weight samples
    weights = torch.softmax(logp0, dim=1)  #(batch_size, Nsample)
    x_0_bar = torch.einsum("bn,bnij->bij", weights, x_0_samples)  # (batch_size, horizon, dims)

    # calculate score
    score = 1 / (1 - alphas_cumprod_t) * (-x_t.to(device) + sqrt_alphas_cumprod_t * x_0_bar)

    return score  # (batch_size, horizon, dims)


def compute_reward(x_0_samples, wall_loc, weight_sdf=0.0, weight_length=0.8, weight_smooth=0.2):
    """
    Compute reward for paths.

    :param points: tensor of shape [batch_size, nsample, horizon, dims], path points
    :param wall_loc: tensor of shape [num_walls, dims], square obstacle centers
    :param weight_sdf: weight for obstacle avoidance
    :param weight_length: weight for path length
    :param weight_smooth: weight for path smoothness
    
    :return: reward tensor of shape [batch_size, nsample]
    """
    device = x_0_samples.device
    batch_size, nsample, horizon, dims = x_0_samples.shape
    sdf_reward = torch.zeros((batch_size, nsample)).to(device)
    length_reward = torch.zeros((batch_size, nsample)).to(device)
    smooth_reward = torch.zeros((batch_size, nsample)).to(device)

    # SDF calculation
    sdf_values = compute_min_sdf(x_0_samples, wall_loc)  # [batch_size, nsample, horizon]
    sdf_reward = sdf_values.sum(dim=-1)  # Sum minimum SDF value, [batch_size, nsample]

    # Path length calculation
    path_diff = torch.diff(x_0_samples, dim=2)  # [batch_size, nsample, horizon-1, dims]
    path_length = torch.norm(path_diff, dim=-1).sum(dim=-1)  # Total path length, [batch_size, nsample]
    length_reward = -path_length  # Shorter path gets higher reward

    # Path smoothness calculation
    path_acceleration = torch.diff(path_diff, dim=2)  # [batch_size, nsample, horizon-2, dims]
    smoothness = torch.norm(path_acceleration, dim=-1).sum(dim=-1)  # Total smoothness, [batch_size, nsample]
    smooth_reward = -smoothness  # Smoother path gets higher reward

    # Combine rewards
    total_reward = (
        weight_sdf * sdf_reward + 
        weight_length * length_reward + 
        weight_smooth * smooth_reward
    )

    return total_reward


def compute_min_sdf(points, wall_center):
    """
    Calculate the minimum signed distance from points to the square defined by wall_center.

    :param points: tensor of shape [batch_size, nsample, horizon, dims=2], path points
    :param wall_center: tensor of shape [num_walls, dims=2], square obstacle centers

    :return: Minimum SDF values, tensor of shape [batch_size, nsample, horizon]
    """
    batch_size, num_walls, dims = wall_center.shape
    square_half_size = 0.5  # Half side length of the square
    delta = torch.abs(points.unsqueeze(-2).expand(-1, -1, -1, num_walls, -1) - wall_center.unsqueeze(1).unsqueeze(1))  # [batch_size, nsample, horizon, num_walls, dims]

    # Inside square: Distance to closest edge
    inside_dist = square_half_size - delta
    inside_dist = torch.min(inside_dist, dim=-1).values  # [batch_size, nsample, horizon, num_walls]

    # Outside square: Distance to the square boundary
    outside_dist = torch.norm(torch.max(delta - square_half_size, torch.zeros_like(delta)), dim=-1)  # [batch_size, nsample, horizon, num_walls]

    # Combine: Positive for outside, negative for inside
    sdf = torch.where((delta <= square_half_size).all(dim=-1), -inside_dist, outside_dist)  # [batch_size, nsample, horizon, num_walls]

    # Directly return the minimum SDF across all obstacles
    min_sdf = sdf.min(dim=-1).values  # [batch_size, nsample, horizon]
    return min_sdf
