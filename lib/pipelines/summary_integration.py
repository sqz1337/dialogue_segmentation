import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from tqdm import tqdm
import nltk
import spacy
import torch
import numpy as np
from scipy.special import softmax
from transformers import LEDTokenizer, LEDForConditionalGeneration
from utilities.dataset import load_dataset_by
from utilities.general import log, embed, calc_metric, get_sentence_encoder
from utilities.tiling import TopicTilingModel, classify_borders

SYNTAX_PARSER = spacy.load('en_core_web_sm')


def _generate_answer(text, model, tokenizer):
  inputs_dict = tokenizer(text, padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to('cuda')
  attention_mask = inputs_dict.attention_mask.to('cuda')
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  summary = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return summary


def _summarize(example, model=None, tokenizer=None):
    text = ' '.join(example['sections'])
    try:
        example['summary'] = _generate_answer(text, model, tokenizer)[0]
    except:
        example['summary'] = '-1'
    return example


def _find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
    return root_token


def _find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        if (token.pos_ == "VERB" and len(ancestors) == 1
                and ancestors[0] == root_token):
            other_verbs.append(token)
    return other_verbs


def _get_clause_token_span_for_verb(verb, doc, all_verbs):
    first_token_index = len(doc)
    last_token_index = 0
    this_verb_children = list(verb.children)
    for child in this_verb_children:
        if child not in all_verbs:
            if child.i < first_token_index:
                first_token_index = child.i
            if child.i > last_token_index:
                last_token_index = child.i
    return first_token_index, last_token_index


def _process_sentence(sentence):
    doc = SYNTAX_PARSER(sentence)
    root = _find_root_of_sentence(doc)
    other_verbs = _find_other_verbs(doc, root)

    token_spans = []
    all_verbs = [root] + other_verbs
    for other_verb in all_verbs:
        if other_verb is None:
            continue
        first_token_index, last_token_index = _get_clause_token_span_for_verb(other_verb, doc, all_verbs)
        token_spans.append((first_token_index, last_token_index))

    sentence_clauses = []
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if start < end:
            clause = doc[start:end]
            sentence_clauses.append(clause)

    sentence_clauses = sorted(sentence_clauses, key=lambda tup: tup[0])
    clauses_text = [clause.text for clause in sentence_clauses]
    return clauses_text


def _calculate_prob_vector(batch):
    prob_vectors = []
    for emb, emb_sum in zip(batch['embeddings'], batch['embeddings_summary']):
        if len(emb_sum) == 0 or len(emb) == 0:
            prob_vectors.append(emb)
            continue
        emb = np.array(emb)
        emb_sum = np.array(emb_sum).T
        logits = np.matmul(emb, emb_sum)
        probs = softmax(logits, axis=1)
        prob_vectors.append(probs)

    batch['probs'] = prob_vectors
    return batch


def split_sent_batched(example):
    doc_summary = example['summary']
    sentences = nltk.tokenize.sent_tokenize(doc_summary, language='english')
    simple_sentences = []
    for sentence in sentences:
        splitted_sentence = _process_sentence(sentence)
        simple_sentences += splitted_sentence

    example['splitted_summary'] = simple_sentences
    return example


def embed_summary(example, model, gpu_batch_size=512):
    lengths = [len(section) for section in example['splitted_summary']]
    all_sentences = []
    for section in example['splitted_summary']:
        for sentence in section:
            all_sentences.append(sentence)
    embeddings = model.encode(all_sentences,
                               show_progress_bar=False,
                               batch_size=gpu_batch_size,
                               convert_to_tensor=True)
    slices = torch.split(embeddings, lengths)
    example['embeddings_summary'] = list(slices)
    return example


# def topic_tilling_batched(batch, tilling_model=None):
#     wds = []
#     pks = []
#     f1s = []

#     for emb, ref in zip(batch['embeddings'], batch['boundaries']):
#         if len(emb) == 0:
#             # pred = '0' * len(ref)
#             pred = tilling_model.transform(emb)
#         else:
#             pred = tilling_model.transform(emb)
#         wd, pk, f1 = calc_metric(ref, pred, refs_passed=True)
#         wds.append(wd)
#         pks.append(pk)
#         f1s.append(f1)

#     batch['wds'] = wds
#     batch['pks'] = pks
#     batch['f1s'] = f1s

#     return batch


def _get_borders_sumseg(example, tiling_model, plot):
    probabilities = example['probs']
    boundaries = tiling_model.transform(probabilities, gold_boundaries=example['boundaries'], plot=plot)
    return boundaries


def get_scores_summary(cfg):
    log(f'Starting to load or create the dataset {cfg.dataset_type}...')
    ds = load_dataset_by(cfg)#.select(list(range(10)))
    
    # from datasets import load_from_disk
    # ds = load_from_disk(cfg.input_path)#.select(list(range(1000)))
    
    log('Dataset generator is built. Loading a sentence-bert model...')
    sentence_model = get_sentence_encoder(cfg)
    # log('Calculating embeddings for the dataset...')
    # ds = ds.map(embed, batched=True, batch_size=cfg.sbert.cpu_batch_size,
    #                      fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})

    # log('Calculating summaries for the dataset...')
    # summary_model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed").to('cuda')
    # summary_tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    # ds = ds.map(_summarize, fn_kwargs={'model': summary_model, 'tokenizer': summary_tokenizer})
    log('Splitting summaries on simple sentences...')
    ds = ds.map(split_sent_batched, num_proc=16)
    log('Calculating embeddings for splitted summaries...')
    ds = ds.map(embed_summary, batched=True, batch_size=cfg.sbert.cpu_batch_size,
                                       fn_kwargs={'model': sentence_model, 'gpu_batch_size': cfg.sbert.gpu_batch_size})
    log('Calculating closeness to summaries...')
    ds = ds.map(_calculate_prob_vector, batch_size=cfg.batch_size,
                                            num_proc=cfg.num_cpu, batched=True)
    ds = ds.with_format('numpy', columns=['embeddings', 'embeddings_summary', 'probs', 'boundaries', 'summary', 'splitted_summary', 'sections'])
    log('Calculating scores...')
    
    if cfg.dataset_type == 'wiki':
        savgol_k = 5 / cfg.mean_segment_length_wiki
    elif cfg.dataset_type == 'ami':
        savgol_k = 5 / cfg.mean_segment_length_ami
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type}!')
    
    # for window_size in [5]:
    #     for threshold in [.7,.8,.9]:
    #         for savgol_k in [savgol_k/5, savgol_k, savgol_k*2, savgol_k*3]:
    #             tiling_model = TopicTilingModel(
    #                 window_size=window_size, 
    #                 threshold=threshold, 
    #                 smoothing_passes=cfg.bertopic.tiling.smoothing_passes, 
    #                 smoothing_window=cfg.bertopic.tiling.smoothing_window,
    #                 n_smooth_savgol=cfg.bertopic.tiling.savgol.n_smooth_savgol, 
    #                 savgol_k=savgol_k,
    #                 polyorder=cfg.bertopic.tiling.savgol.polyorder)
    #             params = [window_size,
    #                     threshold,
    #                     cfg.bertopic.tiling.smoothing_passes, 
    #                     cfg.bertopic.tiling.smoothing_window,
    #                     cfg.bertopic.tiling.savgol.n_smooth_savgol, 
    #                     savgol_k,
    #                     cfg.bertopic.tiling.savgol.polyorder]
        
    tiling_model = TopicTilingModel(
        window_size=cfg.sumseg.tiling.window_size, 
        threshold=cfg.sumseg.tiling.threshold, 
        smoothing_passes=cfg.sumseg.tiling.smoothing_passes, 
        smoothing_window=cfg.sumseg.tiling.smoothing_window,
        n_smooth_savgol=cfg.sumseg.tiling.savgol.n_smooth_savgol, 
        savgol_k=savgol_k,
        polyorder=cfg.sumseg.tiling.savgol.polyorder)
    params = [cfg.sumseg.tiling.window_size,
            cfg.sumseg.tiling.threshold,
            cfg.sumseg.tiling.smoothing_passes, 
            cfg.sumseg.tiling.smoothing_window,
            cfg.sumseg.tiling.savgol.n_smooth_savgol, 
            savgol_k,
            cfg.sumseg.tiling.savgol.polyorder]
    
    print(f'TT params: {params}')

    log('Testing summarization...')
    wds = []
    pks = []
    f1s = []
    boundaries = []
    wd_mean, pk_mean, f1_mean = None, None, None
    
    i = 0
    t = tqdm(ds)
    for example in t:
        # breakpoint()
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
            boundaries.append(_get_borders_sumseg(example, tiling_model, plot=cfg.sumseg.tiling.plot))
        except:
            print('Failed to infer sumseg! Will try to use TopicTiling on embeddings then...')
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
        # breakpoint()

    wd_mean = sum(wds) / len(wds)
    pk_mean = sum(pks) / len(pks)
    f1_mean = sum(f1s) / len(f1s)
    
    log(f'Calculated WD for the dataset {cfg.dataset_type} is {wd_mean:.5f}')
    log(f'Calculated PK for the dataset {cfg.dataset_type} is {pk_mean:.5f}')
    log(f'Calculated F1 for the dataset {cfg.dataset_type} is {f1_mean:.5f}')
        
