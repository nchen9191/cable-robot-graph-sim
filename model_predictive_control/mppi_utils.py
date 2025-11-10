import heapq

import numpy as np
import torch


def scale_ctrl(ctrl, action_lows, action_highs, squash_fn='clamp'):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_highs - action_lows) / 2.0
    act_mid_range = (action_highs + action_lows) / 2.0
    if squash_fn == 'clamp':
        # ctrl = torch.clamp(ctrl, action_lows[0], action_highs[0])
        ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
        return ctrl
    elif squash_fn == 'clamp_rescale':
        ctrl = torch.clamp(ctrl, -1.0, 1.0)
    elif squash_fn == 'tanh':
        ctrl = torch.tanh(ctrl)
    elif squash_fn == 'identity':
        return ctrl
    return act_mid_range.unsqueeze(0) + ctrl * act_half_range.unsqueeze(0)


###########################
## Quasi-Random Sampling ##
###########################

def generate_prime_numbers(num):
    def is_prime(n):
        for j in range(2, ((n // 2) + 1), 1):
            if n % j == 0:
                return False
        return True

    primes = [0] * num  # torch.zeros(num, device=device)
    primes[0] = 2
    curr_num = 1
    for i in range(1, num):
        while True:
            curr_num += 2
            if is_prime(curr_num):
                primes[i] = curr_num
                break

    return primes


def generate_van_der_corput_samples_batch(idx_batch, base):
    inp_device = idx_batch.device
    batch_size = idx_batch.shape[0]
    f = 1.0  # torch.ones(batch_size, device=inp_device)
    r = torch.zeros(batch_size, device=inp_device)
    while torch.any(idx_batch > 0):
        f /= base * 1.0
        r += f * (idx_batch % base)  # * (idx_batch > 0)
        idx_batch = idx_batch // base
    return r


def generate_halton_samples(num_samples, ndims, bases=None, use_ghalton=True, seed_val=123, device=torch.device('cpu'),
                            float_dtype=torch.float64):
    if not use_ghalton:
        samples = torch.zeros(num_samples, ndims, device=device, dtype=float_dtype)
        if not bases:
            bases = generate_prime_numbers(ndims)
        idx_batch = torch.arange(1, num_samples + 1, device=device)
        for dim in range(ndims):
            samples[:, dim] = generate_van_der_corput_samples_batch(idx_batch, bases[dim])
    else:

        # if ndims <= 100:
        #     perms = ghalton.EA_PERMS[:ndims]
        #     sequencer = ghalton.GeneralizedHalton(perms)
        # else:
        sequencer = ghalton.GeneralizedHalton(ndims, seed_val)
        samples = torch.tensor(sequencer.get(num_samples), device=device, dtype=float_dtype)
    return samples


def generate_gaussian_halton_samples(num_samples, ndims, bases=None, use_ghalton=True, seed_val=123,
                                     device=torch.device('cpu'), float_dtype=torch.float64):
    uniform_halton_samples = generate_halton_samples(num_samples, ndims, bases, use_ghalton, seed_val, device,
                                                     float_dtype)

    gaussian_halton_samples = torch.sqrt(torch.tensor([2.0], device=device, dtype=float_dtype)) * torch.erfinv(
        2 * uniform_halton_samples - 1)

    return gaussian_halton_samples


def cost_to_go(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    cost_seq = gamma_seq * cost_seq  # discounted cost sequence
    cost_seq = torch.fliplr(
        torch.cumsum(torch.fliplr(cost_seq), axis=-1))  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq


def snap_to_grid(point, grid_step):
    snapped_x = round(round(point[0] / grid_step) * grid_step, 10)
    snapped_y = round(round(point[1] / grid_step) * grid_step, 10)
    return snapped_x, snapped_y


def snap_to_grid_torch(pt, grid_step, boundaries):
    pt = pt.clone()
    pt[:, 0] = torch.round((pt[:, 0] - boundaries[0]) / grid_step)
    pt[:, 1] = torch.round(-(pt[:, 1] - boundaries[3]) / grid_step)
    return pt.to(torch.int)


def unsnap_to_grid_torch(idx, grid_step, boundaries):
    px = idx[0] * grid_step + boundaries[0]
    py = boundaries[3] - idx[1] * grid_step

    return px, py


def simple_collision(point, obstacles):
    px, py = point
    for xmin, ymin, xmax, ymax in obstacles:
        if (xmin <= px <= xmax) and (ymin <= py <= ymax):
            return True
    return False


def fill_grid(goal, boundary, grid_step=1.0, obstacles=[], device='cpu'):
    assert ((boundary[2] - boundary[0]) / grid_step).is_integer()
    assert ((boundary[3] - boundary[1]) / grid_step).is_integer()

    # BFS from goal
    h_val = {}
    unassigned = set()
    obs_loc = set()
    goal = snap_to_grid(goal, grid_step)
    # h_val[goal]=0
    gaits = [
        [grid_step, 0], [0, grid_step], [0, -grid_step], [-grid_step, 0],
        [grid_step, grid_step], [-grid_step, grid_step], [grid_step, -grid_step], [-grid_step, -grid_step]
    ]

    for i in np.arange(boundary[0], boundary[2] + grid_step, grid_step):
        for j in np.arange(boundary[1], boundary[3] + grid_step, grid_step):
            current = snap_to_grid((i, j), grid_step=grid_step)
            if simple_collision((i, j), obstacles):
                h_val[current] = np.inf
                obs_loc.add(current)
                # all_points.remove()
            else:
                unassigned.add(current)

    open_list = []
    heapq.heappush(open_list, (0, goal))

    while unassigned:
        if not open_list:
            break
        node = heapq.heappop(open_list)

        current = snap_to_grid(node[1], grid_step)
        if current in obs_loc or current not in unassigned:
            continue

        h_val[current] = node[0]
        unassigned.remove(current)

        for k in range(len(gaits)):
            gait_num = gaits[k]
            neighbor = (current[0] + gait_num[0], current[1] + gait_num[1])

            # cost = 1 if k < 4 else np.sqrt(2)
            if k in [1, 2]:
                cost = 3
            elif k > 4:
                cost = np.sqrt(4)
            else:
                cost = 1

            heapq.heappush(open_list, (node[0] + cost, neighbor))

    size_x = int((boundary[2] - boundary[0]) / grid_step) + 1
    size_y = int((boundary[3] - boundary[1]) / grid_step) + 1
    h_val_arr = np.zeros((size_x, size_y), dtype=np.float64)
    obs_val_arr = np.full((size_x, size_y), np.inf, dtype=np.float64)

    for i in np.arange(boundary[0], boundary[2] + grid_step, grid_step):
        for j in np.arange(boundary[1], boundary[3] + grid_step, grid_step):
            i, j = snap_to_grid((i, j), grid_step=grid_step)
            idx_x = round((i - boundary[0]) / grid_step)
            idx_y = round(-(j - boundary[3]) / grid_step)
            h_val_arr[idx_x, idx_y] = h_val[(i, j)]

            if (not simple_collision((i, j), obstacles)
                    and i not in [boundary[0], boundary[2]]
                    and j not in [boundary[1], boundary[3]]):
                dist_to_bound = min([
                    abs(i - boundary[0]),
                    abs(i - boundary[2]),
                    abs(j - boundary[1]),
                    abs(j - boundary[3])
                ])

                dist_to_obs = []
                for xmin, ymin, xmax, ymax in obstacles:
                    if xmin <= i <= xmax:
                        dist_to_obs.append(min(abs(j - ymin), abs(j - ymax)))
                    elif ymin <= j <= ymax:
                        dist_to_obs.append(min(abs(i - xmin), abs(i - xmax)))
                    else:
                        dist_to_obs.append(min([
                            np.sqrt((i - xmin) ** 2 + (j - ymin) ** 2),
                            np.sqrt((i - xmin) ** 2 + (j - ymax) ** 2),
                            np.sqrt((i - xmax) ** 2 + (j - ymin) ** 2),
                            np.sqrt((i - xmax) ** 2 + (j - ymax) ** 2),
                        ]))
                dist_to_obs = min(dist_to_obs) if dist_to_obs else np.inf
                dist = min(dist_to_obs, dist_to_bound) - 1.0
                dist = max(dist, 0.0)
                penalty = 40. / (dist ** 0.25 + 1e-6)

                obs_val_arr[idx_x, idx_y] = penalty

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # plt.imshow(h_val_arr.T)
    # plt.colorbar()
    # plt.title("dist")
    # plt.show()
    #
    # obs_val_arr_cpy = obs_val_arr.copy()
    # # obs_val_arr_cpy[obs_val_arr_cpy == np.inf] = -1.
    # # obs_val_arr_cpy[obs_val_arr_cpy == -1] = obs_val_arr_cpy.max() + 10
    # obs_val_arr_cpy = np.clip(obs_val_arr_cpy, 0., 400)
    #
    # x = np.linspace(boundary[0], boundary[2], h_val_arr.shape[1])
    # y = np.linspace(boundary[1], boundary[3], h_val_arr.shape[0])
    # X, Y = np.meshgrid(y, x)
    #
    # plt.imshow(obs_val_arr_cpy.T)
    # plt.colorbar()
    # plt.title("obs")
    # plt.show()
    # plt.imshow((obs_val_arr_cpy + 3 * h_val_arr).T)
    # plt.colorbar()
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # surf = ax.plot_surface(X, Y, (obs_val_arr_cpy + h_val_arr).T)
    # plt.title("combined")
    # plt.show()

    h_val_arr = torch.from_numpy(h_val_arr).to(device)
    obs_val_arr = torch.from_numpy(obs_val_arr).to(device)
    
    return h_val_arr, obs_val_arr


def heuristic_dir(cost_grid, snapped_pt, num_steps=20):
    snapped_x, snapped_y = snapped_pt.cpu().flatten().numpy().tolist()
    gaits = [
        [1, 0], [0, 1], [0, -1], [-1, 0],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ]

    open_list = [(snapped_x, snapped_y)]
    for n in range(num_steps):
        new_open_list = set()
        for x, y in open_list:
            for g in gaits:
                x_, y_ = x + g[0], y + g[1]
                if 0 <= x_ < cost_grid.shape[0] and 0 <= y_ < cost_grid.shape[1]:
                    new_open_list.add((x_, y_))
        open_list = new_open_list

    best_score, best_pt = 1e9, [0, 0]
    for x, y in open_list:
        score = cost_grid[x, y].item()
        if score < best_score:
            best_score = score
            best_pt = [x, y]

    return best_pt
