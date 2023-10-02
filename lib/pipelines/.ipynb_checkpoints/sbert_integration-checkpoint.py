'''
Run clustering one by one, which means you train a model on each document separately.
'''

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import numpy as np
from datasets import load_from_disk
from utilities.dataset import load_dataset_by
from utilities.general import log, embed, ensure_directory_exists, calc_metric, get_sentence_encoder
from utilities.tiling import TopicTilingModel, classify_borders
from utilities.clustering import BERTopicModel
from tqdm import tqdm


def _get_borders_clustering(example, models_gen):
    topic_model, tiling_model = models_gen()
    
    text = example['sections']
    embeddings = np.array(example['embeddings'])
    
    topic_model.fit(text, embeddings)
    probabilities = topic_model.transform(text, embeddings)
    
    boundaries = tiling_model.transform(probabilities)
    
    # topic_ids, all_topics = topic_model.get_topics(probabilities)
    # assert len(topic_ids) == len(embeddings)
    
    return boundaries


def embed_and_save(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    ds = load_dataset_by(cfg)
    log('Dataset generator is built. Loading a sentence-bert model...')
    sentence_model = get_sentence_encoder(cfg)
    log('Calculating embeddings for the dataset...')
    ds_embedded = ds.map(embed, batched=True, batch_size=cfg.sbert.cpu_batch_size,
                         fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})
    ensure_directory_exists(cfg.output_path)
    ds_embedded.save_to_disk(cfg.output_path)
    

def get_scores_sentence_bert(cfg):
    log(f'Loading the dataset {cfg.dataset_type} from disk...')
    ds_embedded = load_from_disk(cfg.input_path)#.select(list(range(1000)))
    log('Dataset is loaded.')
    
    tiling_model = TopicTilingModel(window_size=cfg.bertopic.tiling.window_size, 
                                        threshold=cfg.bertopic.tiling.threshold, 
                                        smoothing_passes=cfg.bertopic.tiling.smoothing_passes, 
                                        smoothing_window=cfg.bertopic.tiling.smoothing_window)
    
    def get_models():
        topic_model = BERTopicModel(cfg)
        return topic_model, tiling_model
        
    # for window_size in [3,5]:
    #     for threshold in [.7,.9]:
    #         for smoothing_passes in [0,1,3]:
    #             params = [window_size,
    #                     threshold]
    #             print(f'TT params: {params}')
            
    log('Making a segmentation...')
    wds = []
    pks = []
    f1s = []
    boundaries = []
    wd_mean, pk_mean, f1_mean = None, None, None
    
    t = tqdm(ds_embedded)
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
        
        if cfg.clustering:
            try:
                boundaries.append(_get_borders_clustering(example, get_models))
            except:
                boundaries.append(classify_borders(example['embeddings'], 
                                            window_size=cfg.tiling_window_size,
                                            threshold=cfg.tiling_threshold, 
                                            smoothing_passes=cfg.smoothing_passes,
                                            smoothing_window=cfg.smoothing_window))
        else:
            boundaries.append(classify_borders(example['embeddings'], 
                                            window_size=cfg.tiling_window_size,
                                            # window_size=window_size,
                                            threshold=cfg.tiling_threshold, 
                                            # threshold=threshold,
                                            smoothing_passes=cfg.smoothing_passes,
                                            # smoothing_passes=smoothing_passes,
                                            smoothing_window=cfg.smoothing_window))

        wd, pk, f1 = calc_metric(example, boundaries[-1])
        wds.append(wd)
        pks.append(pk)
        f1s.append(f1)
        
    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)
    log(f'Calculated WD for the dataset {cfg.dataset_type} is {wd_mean:.5f}')
    log(f'Calculated PK for the dataset {cfg.dataset_type} is {pk_mean:.5f}')
    log(f'Calculated F1 for the dataset {cfg.dataset_type} is {f1_mean:.5f}')
