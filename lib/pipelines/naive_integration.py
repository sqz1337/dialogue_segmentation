import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import random
from tqdm import tqdm

from utilities.dataset import get_boundaries, load_dataset_by, calculate_statistics
from utilities.general import log, calc_metrics, get_mean_metrics


def _get_borders_random(example, mean_segment_length=None, seed=123, *args, **kwargs):
    random.seed = seed
    assert mean_segment_length is not None
    
    corpus = example['sections']
    p = 1 / mean_segment_length
    topics = [0]
    topic_id = 0
    
    for _ in range(1, len(corpus)):
        if random.uniform(0, 1) <= p:
            topic_id += 1
        topics.append(topic_id)
            
    assert len(topics) == len(corpus)
    boundaries = get_boundaries(topics)
    return boundaries


def _get_borders_all_zeros(example, *args, **kwargs):
    corpus = example['sections']
    boundaries = '0' * len(corpus)
    return boundaries


def _get_borders_all_ones(example, *args, **kwargs):
    corpus = example['sections']
    boundaries = '1' * len(corpus)
    return boundaries
    
    
def get_scores_naive(cfg):
    log(f'Loading the dataset {cfg.dataset_type} from disk...')
    ds = load_dataset_by(cfg)
    
    # from datasets import load_from_disk
    # ds = load_from_disk(cfg.input_path)
    
    log('Dataset is loaded.')
    
    mean_segment_length = calculate_statistics(ds)
    log(f'mean_segment_length for dataset {cfg.dataset_type} is {mean_segment_length}')
    
    if cfg.dataset_type == 'wiki':
        mean_segment_length = cfg.mean_segment_length_wiki
    elif cfg.dataset_type == 'ami':
        mean_segment_length = cfg.mean_segment_length_ami
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type} exist!')
    
    log('Making a segmentation...')
    for get_borders_function, name in zip(
        [_get_borders_random, _get_borders_all_zeros, _get_borders_all_ones],
        ['random_segmentation', 'all_zeros_segmentation', 'all_ones_segmentation']):
        boundaries = []
        t = tqdm(ds)
        for example in t:
            boundaries.append(get_borders_function(example, mean_segment_length=mean_segment_length))
            
        log(f'Calculating the metrics for {name}...')
        wds, pks, f1s = calc_metrics(ds, boundaries)
        wd_mean, pk_mean, f1_mean = get_mean_metrics(wds, pks, f1s)
        log(f'Calculated WD for the dataset {cfg.dataset_type} is {wd_mean:.5f}.')
        log(f'Calculated PK for the dataset {cfg.dataset_type} is {pk_mean:.5f}.')
        log(f'Calculated F1 for the dataset {cfg.dataset_type} is {f1_mean:.5f}.')
