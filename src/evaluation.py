import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils2 import move_to
from utils2.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils2.functions import parse_softmax_temperature, torch_load_cpu, _load_model_file
from problems.tsppc.problem_tsppc import TSPPC
from nets.attention_model import AttentionModel
mp = torch.multiprocessing.get_context('spawn')

def load_model(path, opts):
    model_filename = path
    problem = TSPPC()

    load_data = torch_load_cpu(model_filename)

    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        problem=problem,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization='batch',
        tanh_clipping=8,
        checkpoint_encoder=False,
        shrink_size=None,
        shared_keyvalues='embedder.layers.1.mha.module.W_K_ps' not in load_data.get('model') and 
                         'embedder.layers.1.mha.module.W_K_sp' not in load_data.get('model') and
                         'embedder.layers.1.mha.module.W_K_pg' not in load_data.get('model'),
        pred2succ ='embedder.layers.1.mha.module.W_Q_ps' in load_data.get('model'),
        succ2pred ='embedder.layers.1.mha.module.W_Q_sp' in load_data.get('model'),
        pred_group='embedder.layers.1.mha.module.W_Q_pg' in load_data.get('model'),
        sparse_thresh = None if load_data.get('additional_model_params') is None
                        else load_data.get('additional_model_params')["sparse_thresh"]
    )
    
    # Overwrite model parameters by parameters to load
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model, not opts.quite)

    model.eval()  # Put in eval mode

    return model, None


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model, opts)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model, opts)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    if not opts.quite:
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])

    if not opts.no_output_file:
        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
                dataset_basename, model_name,
                opts.decode_strategy,
                width if opts.decode_strategy != 'greedy' else '',
                softmax_temp, opts.offset, opts.offset + len(costs), ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        save_dataset((results, parallelism), out_file)

    if not opts.quite:
        print('Model: ', opts.model.split(os.path.sep)[-1])
        total_params = sum(p.numel() for p in model.parameters())
        print('Total parameters: ', total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: ', trainable_params)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            elif model.problem.NAME == "tsppc":
                seq = np.trim_zeros(seq).tolist() + [0]
            else:
                seq = None
                # assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))

    return results


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--problem', default='tsppc', help="The problem to solve, default 'tsppc'")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--no_output_file',action='store_true', help="Prevent to save output as a file")
    parser.add_argument('--quite',action='store_true', help="Prevent printing during evaluation")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()
    opts.no_progress_bar = opts.no_progress_bar or opts.quite

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    if opts.width is None:
        opts.width = [0]

    return opts



if __name__ == "__main__":
    opts = get_options()

    for width in opts.width:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
