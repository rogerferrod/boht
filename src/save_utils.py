import os
import torch
import pickle


def save_model(model, path, step, max_keep=5):
    """
    Dump the model, keeping last 'max_keep' savings
    """

    save_path = os.path.join(path, 'checkpoint')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = f'model_{step}.pt'
    checkpoint_file = os.path.join(save_path, 'checkpoint_file.pkl')
    _save_checkpoint_file(checkpoint_file, filename, save_path, max_keep)
    model.close_writer()
    torch.save(model, os.path.join(save_path, filename))


def load_model(path):
    checkpoint = torch.load(path)
    return checkpoint


def _save_checkpoint_file(checkpoint_file, filename, path, max_keep):
    chkpoint = []

    # 1 - check if checkpoint file exist
    if os.path.exists(checkpoint_file):
        chkpoint = _read_checkpoint_file(checkpoint_file)

    # 2 - check if exceed
    len_checkpoints = len(chkpoint)
    if len_checkpoints >= max_keep:
        to_delete = chkpoint[0]
        os.remove(os.path.join(path, to_delete))
        chkpoint = chkpoint[1:]

    # 3 - update chkpoint
    chkpoint.append(filename)

    # 4 - save file
    with open(checkpoint_file, 'wb') as fout:
        pickle.dump(chkpoint, fout, protocol=pickle.DEFAULT_PROTOCOL)


def _read_checkpoint_file(checkpoint_file):
    with open(checkpoint_file, 'rb') as fin:
        chkpoint = pickle.load(fin)
        return chkpoint
