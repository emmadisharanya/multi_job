#!/usr/bin/env python
"""
Multi-Job Federated Learning — RLDS Scheduler
==============================================
Reproduces the experimental setup of:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022  (arXiv:2112.05928)

Identical setup to run_random_improved.py — only the scheduler differs.
See that file for full hyperparameter commentary.
"""

import os
import sys
import time
import random
import json

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet  import ResNet18
from models.lenet   import LeNet5
from models.alexnet import AlexNet
from federated.client import FLClient
from federated.server import FLServer
from data.non_iid_partition import create_non_iid_datasets
from schedulers.multijob_rlds import MultiJobRLDSScheduler

# ── Live plot support ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

_fig = _axes = None
_acc_history  = {0: [], 1: [], 2: []}
_loss_history = {0: [], 1: [], 2: []}
_round_history = {0: [], 1: [], 2: []}
_JOB_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
_JOB_NAMES  = ['ResNet18+CIFAR10', 'LeNet5+MNIST', 'AlexNet+MNIST']

def setup_plots():
    global _fig, _axes
    if not PLOT_AVAILABLE:
        return
    plt.ion()
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig.suptitle('RLDS Scheduler — Training Progress', fontsize=13)
    for ax, title, ylabel in [
        (_axes[0], 'Test Accuracy per Round',  'Accuracy (%)'),
        (_axes[1], 'Test Loss per Round',       'Loss'),
    ]:
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def update_plots(job_id, round_num, acc, loss):
    """Call at the end of every round for each active job."""
    _acc_history[job_id].append(acc)
    _loss_history[job_id].append(loss)
    _round_history[job_id].append(round_num)

    if not PLOT_AVAILABLE or _axes is None:
        return

    _axes[0].cla(); _axes[1].cla()
    for j in range(3):
        if _round_history[j]:
            _axes[0].plot(_round_history[j], _acc_history[j],
                          color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
            _axes[1].plot(_round_history[j], _loss_history[j],
                          color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)

    for ax, ylabel in [(_axes[0], 'Accuracy (%)'), (_axes[1], 'Loss')]:
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    _axes[0].set_title('Test Accuracy per Round')
    _axes[1].set_title('Test Loss per Round')
    _fig.canvas.draw()
    _fig.canvas.flush_events()

def save_plots(out_dir='/kaggle/working'):
    if not PLOT_AVAILABLE or _fig is None:
        return
    plt.ioff()
    path = os.path.join(out_dir, 'rlds_training_curves.png')
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved → {path}')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('MULTI-JOB RLDS SCHEDULER  (paper: Zhou et al. AAAI-22)')
    print('=' * 70)

    SEED = 42
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice : {device}')

    # ── Hyperparameters — identical to Random baseline ────────────────────────
    NUM_DEVICES       = 100
    DEVICES_PER_ROUND = 10
    LOCAL_EPOCHS      = 5
    BATCH_SIZE        = 64
    LEARNING_RATE     = 0.01
    MAX_ROUNDS        = 800

    # ── Paper cost-model weights (Formula 2 in paper) ────────────────────────
    ALPHA = 0.3   # weight of time cost
    BETA  = 0.7   # weight of fairness cost

    # ── Job configuration — SAME as Random baseline ───────────────────────────
    JOBS = {
        0: ('resnet18', 'cifar10', 3, 32, 52.0),
        1: ('lenet5',   'mnist',   1, 28, 98.0),
        2: ('alexnet',  'mnist',   1, 28, 99.0),
    }
    JOB_NAMES = {
        0: 'ResNet18 + CIFAR-10',
        1: 'LeNet-5  + MNIST',
        2: 'AlexNet  + MNIST',
    }

    print('\nJobs:')
    for j, (m, d, _, _, t) in JOBS.items():
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  target={t}%')

    # ── Datasets ─────────────────────────────────────────────────────────────
    print('\nCreating non-IID datasets (2 classes/device)...')
    job_client_datasets = {}
    job_test_datasets   = {}
    for j, (_, dataset, _, _, _) in JOBS.items():
        client_data, test_data = create_non_iid_datasets(
            dataset, NUM_DEVICES, num_classes_per_device=2, seed=SEED + j
        )
        job_client_datasets[j] = client_data
        job_test_datasets[j]   = test_data
        print(f'  Job {j} ({dataset}): {len(test_data)} test samples')

    # ── Models & servers ─────────────────────────────────────────────────────
    print('\nInitialising models...')
    servers = {}
    for j, (model_key, _, ch, sz, _) in JOBS.items():
        if model_key == 'resnet18':
            model = ResNet18(num_classes=10, input_channels=ch)
        elif model_key == 'lenet5':
            model = LeNet5(num_classes=10)
        elif model_key == 'alexnet':
            model = AlexNet(num_classes=10, input_channels=ch, input_size=sz)
        else:
            raise ValueError(f'Unknown model: {model_key}')
        servers[j] = FLServer(model.to(device), job_test_datasets[j], device=str(device))
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  params={n_params:,}')

    # ── Clients ───────────────────────────────────────────────────────────────
    print(f'\nCreating {NUM_DEVICES} clients per job...')
    clients = {}
    device_caps = {}
    rng = np.random.default_rng(SEED)
    for d in range(NUM_DEVICES):
        # Device capability drawn from the shift-exponential model (paper Eq. 4)
        # a_k ~ Uniform(0.5, 2.0),  μ_k ~ Uniform(0.1, 1.0)
        device_caps[d] = {
            'capability':  float(rng.uniform(0.5, 2.0)),
            'fluctuation': float(rng.uniform(0.1, 1.0)),
        }
    for j in range(3):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d], BATCH_SIZE, LEARNING_RATE, device)
            for d in range(NUM_DEVICES)
        }

    # ── RLDS Scheduler ────────────────────────────────────────────────────────
    print('\nInitialising RLDS scheduler...')
    scheduler = MultiJobRLDSScheduler(
        num_devices=NUM_DEVICES,
        devices_per_round=DEVICES_PER_ROUND,
        num_jobs=3,
        device_capabilities=device_caps,
        alpha=ALPHA,
        beta=BETA,
        epsilon=0.3,     # ε-greedy exploration rate
        gamma=0.9,       # baseline EMA decay
        learning_rate=1e-3,
    )
    print('  RLDS ready')

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    setup_plots()   # ◄── live plots initialised HERE (before loop)

    job_done      = {0: False, 1: False, 2: False}
    job_times     = {0: 0.0,   1: 0.0,   2: 0.0}
    job_rounds    = {0: 0,     1: 0,     2: 0}
    job_final_acc = {0: 0.0,   1: 0.0,   2: 0.0}

    log = {j: {'rounds': [], 'acc': [], 'loss': []} for j in range(3)}

    global_start = time.time()
    round_num    = 0

    with tqdm(total=MAX_ROUNDS, desc='RLDS') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1
            round_start = time.time()

            # ── RLDS device assignment ────────────────────────────────────────
            assignments = scheduler.select_devices_for_all_jobs()

            round_accs = []

            for j in range(3):
                if job_done[j]:
                    round_accs.append(JOBS[j][4])
                    continue

                selected = assignments[j]

                # ── Local training + FedAvg aggregation ──────────────────────
                local_updates = [
                    clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                    for d in selected
                ]
                servers[j].aggregate(local_updates)
                job_rounds[j] += 1

                # ── Evaluate ──────────────────────────────────────────────────
                result = servers[j].evaluate()
                acc  = result['test_accuracy']
                loss = result['test_loss']
                job_final_acc[j] = acc
                round_accs.append(acc)

                # ── ▼▼▼ LIVE PLOT UPDATE — called every round ▼▼▼ ────────────
                update_plots(j, job_rounds[j], acc, loss)
                # ── ▲▲▲ END PLOT UPDATE ▲▲▲ ──────────────────────────────────

                log[j]['rounds'].append(job_rounds[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                # ── Check target ──────────────────────────────────────────────
                target = JOBS[j][4]
                if acc >= target and not job_done[j]:
                    job_done[j] = True
                    job_times[j] = time.time() - global_start
                    print(f'\n  ✓ Job {j} ({JOB_NAMES[j]}) reached {target}%'
                          f' at round {job_rounds[j]}'
                          f' ({job_times[j]/60:.1f} min)')

            # ── RLDS policy update ────────────────────────────────────────────
            time_cost = time.time() - round_start
            scheduler.update_selections(assignments)
            reward = scheduler.compute_reward(round_accs, time_cost)
            scheduler.update_policy(reward, assignments)

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(3) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()   # ◄── save final plot

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = '/kaggle/working/rlds_log.json'
    try:
        with open(log_path, 'w') as f:
            json.dump(log, f)
        print(f'  Log saved → {log_path}')
    except Exception:
        pass

    # ── Fairness stats ────────────────────────────────────────────────────────
    stats = scheduler.get_stats()

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = sum(job_times.values())
    print('\n' + '=' * 70)
    print('RLDS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<26} {"Time (min)":>12} {"Rounds":>8} {"Acc":>8}')
    print('-' * 70)
    for j in range(3):
        print(f'  {j}    {JOB_NAMES[j]:<26} '
              f'{job_times[j]/60:>10.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%')
    print('-' * 70)
    print(f'  Total time: {total_time/60:.1f} min  ({total_time/3600:.2f} h)')

    print('\nFairness (σ of per-device selection counts):')
    for j in range(3):
        print(f'  Job {j}: σ={stats[j]["std"]:.2f}  '
              f'min={stats[j]["min"]}  max={stats[j]["max"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()