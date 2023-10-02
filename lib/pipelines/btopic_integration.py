'''
Run clustering by training on train part of a dataset, and then get results on test part of the dataset.
'''

import os
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import numpy as np
from datasets import load_from_disk
from utilities.general import log, calc_metric
from utilities.tiling import TopicTilingModel, classify_borders
from utilities.clustering import BERTopicModel
from tqdm import tqdm


def _get_borders_clustering(example, topic_model, tiling_model, plot):
    text = example['sections']
    embeddings = np.array(example['embeddings'])
    
    probabilities = topic_model.transform(text, embeddings)
    
    boundaries = tiling_model.transform(probabilities, gold_boundaries=example['boundaries'], plot=plot)
    
    # topic_ids, all_topics = topic_model.get_topics(probabilities)
    # assert len(topic_ids) == len(embeddings)
    
    return boundaries


def get_scores_bert_topic(cfg):
    log(f'Working with dataset {cfg.train_path}')
    
    bt_path = cfg.bertopic.pickle_path
    topic_model = BERTopicModel(cfg, log_status=True, checkpoints_path=bt_path)
    if bt_path is not None and os.path.exists(bt_path):
        log(f'Loading BERTopic from a checkpoint {bt_path}...')
        topic_model.load(bt_path)
    else:
        log('No BERTopic checkpoints found.')
        log(f'Loading train dataset {cfg.dataset_type} from disk...')
        train_embedded = load_from_disk(cfg.train_path) \
            .with_format('numpy', columns=['embeddings', 'sections'])#.select(list(range(50000)))
        log('Preparing train text and embeddings...')
        embeddings_train = []
        text_train = []
        n_leave = cfg.bertopic.n_leave
        all_embeddings = train_embedded['embeddings']
        all_text = train_embedded['sections']
        for i in tqdm(range(len(all_embeddings))):
            n = all_embeddings[i].shape[0]
            step = n // n_leave if n_leave else 0
            indices_to_select = list(range(0, n, step + 1))
            embeddings_train.append(all_embeddings[i][indices_to_select, :])
            text_train.append(all_text[i][indices_to_select])
        embeddings_train = np.concatenate(embeddings_train)
        text_train = np.concatenate(text_train)
        log('Training BERTopic...')
        topic_model.fit(text_train, embeddings_train)
        if bt_path is not None:
            log(f'Saving BERTopic to {bt_path}...')
        
        del train_embedded
        del embeddings_train
        del text_train
            
    if cfg.dataset_type == 'wiki':
        savgol_k = cfg.mean_segment_length_wiki
    elif cfg.dataset_type == 'ami':
        savgol_k = 10 / cfg.mean_segment_length_ami
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type}!')
    
    log(f'Loading test dataset {cfg.dataset_type} from disk...')
    test_embedded = load_from_disk(cfg.test_path)#.select(list(range(500)))
    t = tqdm(test_embedded)
    
    # for window_size in [4]:
    #     for threshold in [.5]:
    #         tiling_model = TopicTilingModel(
    #             window_size=window_size, 
    #             threshold=threshold, 
    #             smoothing_passes=cfg.bertopic.tiling.smoothing_passes, 
    #             smoothing_window=cfg.bertopic.tiling.smoothing_window,
    #             n_smooth_savgol=cfg.bertopic.tiling.savgol.n_smooth_savgol, 
    #             savgol_k=savgol_k,
    #             polyorder=cfg.bertopic.tiling.savgol.polyorder)
    #         params = [window_size,
    #                 threshold,
    #                 cfg.bertopic.tiling.smoothing_passes, 
    #                 cfg.bertopic.tiling.smoothing_window,
    #                 cfg.bertopic.tiling.savgol.n_smooth_savgol, 
    #                 savgol_k,
    #                 cfg.bertopic.tiling.savgol.polyorder]

    tiling_model = TopicTilingModel(
        window_size=cfg.bertopic.tiling.window_size, 
        threshold=cfg.bertopic.tiling.threshold, 
        smoothing_passes=cfg.bertopic.tiling.smoothing_passes, 
        smoothing_window=cfg.bertopic.tiling.smoothing_window,
        n_smooth_savgol=cfg.bertopic.tiling.savgol.n_smooth_savgol, 
        savgol_k=savgol_k,
        polyorder=cfg.bertopic.tiling.savgol.polyorder)
    params = [cfg.bertopic.tiling.window_size,
            cfg.bertopic.tiling.threshold,
            cfg.bertopic.tiling.smoothing_passes, 
            cfg.bertopic.tiling.smoothing_window,
            cfg.bertopic.tiling.savgol.n_smooth_savgol, 
            savgol_k,
            cfg.bertopic.tiling.savgol.polyorder]
    
    print(f'TT params: {params}')
                        
    log('Testing BERTopic...')
    wds = []
    pks = []
    f1s = []
    boundaries = []
    wd_mean, pk_mean, f1_mean = None, None, None
    
    i = 0
    for example in t:
        if i > 0:
            wd_mean = sum(wds) / len(wds)
            pk_mean = sum(pks) / len(pks)
            f1_mean = sum(f1s) / len(f1s)
            
            description = f'wd: {wd_mean:.3f}, '
            description += f'pk: {pk_mean:.3f}, '
            description += f'f1: {f1_mean:.3f}'
            t.set_description(description)
        i+=1
        
        try:
            boundaries.append(_get_borders_clustering(example, topic_model, tiling_model, plot=cfg.bertopic.tiling.plot))
        except:
            print('Failed to infer bertopic! Will try to use TopicTiling on embeddings then...')
            boundaries.append(classify_borders(example['embeddings'], 
                                        window_size=cfg.tiling_window_size,
                                        threshold=cfg.tiling_threshold, 
                                        smoothing_passes=cfg.smoothing_passes,
                                        smoothing_window=cfg.smoothing_window))
            
        wd, pk, f1 = calc_metric(example, boundaries[-1])
        wds.append(wd)
        pks.append(pk)
        f1s.append(f1)
        
        # input()

    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)
    
    log(f'Calculated WD for the dataset {cfg.dataset_type} is {wd_mean:.5f}')
    log(f'Calculated PK for the dataset {cfg.dataset_type} is {pk_mean:.5f}')
    log(f'Calculated F1 for the dataset {cfg.dataset_type} is {f1_mean:.5f}')
        