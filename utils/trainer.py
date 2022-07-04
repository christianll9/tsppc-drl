import os
import sys
import math
import uuid
import utils.problem_generator as probgen

file_dir = os.path.dirname(os.path.abspath(__file__))


def kool2018(n_nodes:int=20, n_epochs:int=100, epoch_size:int=100000, baseline:str="rollout", batch_size:int=200,
    warm_load_path:str=None, run_name:str=None, val_size:int=10000, eval_batch_size=100, progress_bar=True, distribution="uniform",
    sparse_thresh=math.inf):
    """
    No heterogeneous attentions. The trained model can be used in `solver.model_solve_single`
    """
    if run_name is None: run_name = f"tsppc-kool2018-n{n_nodes}-e{epoch_size}-{uuid.uuid4()}"
    start_trainer(probgen.preprocessed_tsppcs, n_nodes, n_epochs, epoch_size, baseline, batch_size,
        warm_load_path, run_name, val_size, eval_batch_size, progress_bar, distr=distribution,
        sparse_thresh=sparse_thresh, shared_keyvalues=False, pred2succ=False, succ2pred=False, pred_group=False)


def train(n_nodes:int=20, n_epochs:int=100, epoch_size:int=100000, baseline:str="rollout", batch_size:int=200,
    warm_load_path:str=None, run_name:str=None, val_size:int=10000, eval_batch_size=100, progress_bar=True, distribution="uniform",
    sparse_thresh=math.inf, shared_keyvalues=False, pred2succ=True, succ2pred=True, pred_group=False):
    """
    Training wrapper for our proposed model
    To further train a pretrained model, you can specify the path to the model in `warm_load_path`.
    pred2succ => attentions from predecessor to successor
    succ2pred => attentions from successor to predecessor
    pred_group => attentions to all members of a precedence chain group
    """
    if run_name is None: run_name = f"tsppc-n{n_nodes}-e{epoch_size}-{uuid.uuid4()}"
    start_trainer(probgen.preprocessed_tsppcs, n_nodes, n_epochs, epoch_size, baseline, batch_size,
        warm_load_path, run_name, val_size, eval_batch_size, progress_bar, distr=distribution,
        sparse_thresh=sparse_thresh, shared_keyvalues=shared_keyvalues, pred2succ=pred2succ, succ2pred=succ2pred, pred_group=pred_group)


def start_trainer(val_data_generator, n_nodes:int, n_epochs:int, epoch_size:int, baseline:str, batch_size:int,
    warm_load_path:str, run_name:str, val_size:int, eval_batch_size=100, progress_bar=True, distr="uniform",
    sparse_thresh=math.inf, shared_keyvalues=False, pred2succ=True, succ2pred=True, pred_group=False):

    cwd = os.getcwd() #current working directory
    tmp_file_path = os.path.join(cwd, f"val_tmp_{uuid.uuid4()}.pkl")
    print(f"Generate validation dataset for training  ({val_size} instances in {tmp_file_path})")
    val_data_generator(n_problems=val_size, output_path=tmp_file_path, n_nodes=n_nodes, distr=distr)

    sys.argv = ['run_training.py', "--problem", "tsppc", '--baseline', baseline, '--run_name', run_name, '--batch_size', str(batch_size),
                '--graph_size', str(n_nodes), '--n_epochs', str(n_epochs), '--epoch_size', str(epoch_size),
                "--data_distribution", distr, "--val_dataset", tmp_file_path, '--eval_batch_size', str(eval_batch_size),
                "--sparse_thresh", str(sparse_thresh)
    ]
    
    if warm_load_path is not None: sys.argv += ["--load_path", warm_load_path]
    if not progress_bar: sys.argv += ["--no_progress_bar"]
    if shared_keyvalues: sys.argv += ["--shared_keyvalues"]
    if pred2succ: sys.argv += ["--pred2succ"]
    if succ2pred: sys.argv += ["--succ2pred"]
    if pred_group: sys.argv += ["--pred_group"]

    sys.path.append(os.path.join(file_dir, "..", "src"))
    import run_training
    import options
    print(" ".join(sys.argv))
    run_training.run(options.get_options())

    os.remove(tmp_file_path) # remove val_dataset
    
