from typing import List
import torch
from torch.optim import lr_scheduler, Adam
from torch import nn
import numpy as np
from environments.environment_abstract import Environment, State
from time import time

from utils import misc_utils

BATCH_SIZE = 250

def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    return nn.Sequential(
                nn.Linear(81,BATCH_SIZE).to(torch.float32),
                nn.ReLU(),
                nn.Linear(BATCH_SIZE, BATCH_SIZE).to(torch.float32),
                nn.ReLU(),
                nn.Linear(BATCH_SIZE, BATCH_SIZE).to(torch.float32),
                nn.ReLU(),
                nn.Linear(BATCH_SIZE, 1).to(torch.float32))


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    print_skip = 100

    loss_fn = nn.MSELoss()
    optimizer = Adam(nnet.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.99,patience=100)
    batch_start_idx = 0
    batch_start_time = time()

    for itr in range(train_itr, train_itr + num_itrs):
        # get batch of training examples
        start_idx = batch_start_idx
        end_idx = batch_start_idx + batch_size
        input_batch = torch.tensor(states_nnet[start_idx:end_idx], dtype=torch.float32)
        target_batch = torch.tensor(outputs[start_idx:end_idx], dtype=torch.float32)

        # complete pass over batch
        pred_batch = nnet(input_batch)
        loss = loss_fn(target_batch, pred_batch)
        if itr % print_skip == 0:  # print loss every 100 training iterations
            print(f"Itr: {itr}, "
                  f"lr: {optimizer.param_groups[0]['lr']:.3e}, "
                  f"loss: {round(loss.item(), 5)}, "
                  f"targ_ctg: {round(target_batch.float().mean().item(), 2)}, "
                  f"nnet_ctg: {round(pred_batch.float().mean().item(), 2)}, "
                  f"Time: {round(time() - batch_start_time, 2)}")
            batch_start_time = time()
        # update lr
        scheduler.step(loss)
        loss.backward()


        # update optimizer
        optimizer.step()
        optimizer.zero_grad()

        # increment to next batch
        batch_start_idx += batch_size


def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    states_exp, tc_l = env.expand(states)
    batches_count = len(states_exp) // BATCH_SIZE

    output = []

    with torch.no_grad():
        for i in range(batches_count):
            states_exp_flat, states_split_idxs = misc_utils.flatten(states_exp[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
            states_exp_flat_tensor = torch.tensor(env.state_to_nnet_input(states_exp_flat), dtype=torch.float32)
            tmp = nnet(states_exp_flat_tensor)
            output.extend(misc_utils.unflatten(tmp.tolist(), states_split_idxs))
    
    J = np.array(output).squeeze()
    g = np.array(tc_l)

    is_goal = env.is_solved(states)
    J_prime = ((g+J).min(1)) * np.logical_not(is_goal)

    return J_prime.tolist()
