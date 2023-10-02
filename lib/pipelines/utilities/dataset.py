from pathlib2 import Path
import codecs
import json
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from .wiki_loader import WikipediaDataSet


def get_boundaries(labels):
    assert len(labels) > 1
    boundaries = ['0']
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append('1')
        else:
            boundaries.append('0')
    return ''.join(boundaries)


def get_labels(indices):
    max_length = indices[-1] + 1
    labels = []
    label = 0
    for i in range(max_length):
        labels.append(label)
        if i + 1 in indices[:-1]:
            label += 1
    return labels


# Wiki:

def get_files(path):
    'Ref: https://github.com/koomri/text-segmentation'
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


class WikiDataset:
    def __init__(self, root):
        self.dataset = WikipediaDataSet(root, None)

    def _get_sample(self):
        for i in range(self.dataset.__len__()):
            sections, targets, path = self.dataset.__getitem__(i)
            if len(sections) <= 1 or len(targets) <= 1:
                continue
            else:
                labels = get_labels(targets)
                boundaries = get_boundaries(labels)
                assert len(sections) == len(labels)
                assert len(sections) == len(boundaries)
                
                output = {'path': str(path),
                            'sections': sections,
                            'labels': labels,
                            'boundaries': boundaries,
                            'split_indices': targets,
                            'topic_names': None}
                
                yield output

    def get_generator(self):
        return self._get_sample
    
    
# AMI:

def load_json(file_path):
    with codecs.open(file_path, "r", "utf-8") as f:
        datas = json.load(f)
    print("Load {} finished, Data size:{}".format(file_path.split("/")[-1], len(datas)))
    return datas


def preprocess(text):
    # filter some noises caused by speech recognition
    def clean_data(text_):
        text_ = text_.replace('<vocalsound>', '')
        text_ = text_.replace('<disfmarker>', '')
        text_ = text_.replace('a_m_i_', 'ami')
        text_ = text_.replace('l_c_d_', 'lcd')
        text_ = text_.replace('p_m_s', 'pms')
        text_ = text_.replace('t_v_', 'tv')
        text_ = text_.replace('<pause>', '')
        text_ = text_.replace('<nonvocalsound>', '')
        text_ = text_.replace('<gap>', '')
        return text_
    
    text = clean_data(text)
    
    fillers = ["um", "uh", "oh", "hmm", "you know", "like"]
    fillers += [filler + " " for filler in fillers]  # filler inside caption with other words
    fillers = [re.compile(f"(?i){filler}") for filler in fillers]  # make it case-insensitive

    for filler in fillers:
        text = filler.sub("", text)

    # captions_with_multiple_sentences = text.count(".")
    # if captions_with_multiple_sentences > 0:
    #     print(f"WARNING: Found {captions_with_multiple_sentences} captions with "
    #           "multiple sentences; sentence embeddings may be inaccurate.", file=sys.stderr)

    if len(text) <= 20:
        return None
    
    text = text.strip()
    text = ' '.join(text.split())

    return text
    
    
class AMIDataset:
    def __init__(self, root):
        '''Set topic_key='id' if you want to set each section to different topic_id, and 
        set topic_key='topic' if you want to preserve topic names (which will imply that topics can be
        the same for different section, so it can 'come back')
        '''
        self.textfiles = get_files(root)
        self.topic_key = 'id'

    def _get_sections(self, segments):
        sections = []  # text in each section
        labels = []  # topic id for each section
        topic_ids = {}  # topic name : topic id

        for segment in segments:
            topic_name = segment[self.topic_key]

            if segment['dialogueacts'] != 'None' and segment['subtopics'] == 'None':
                dialogue = []
                starttimes = []
                for d in segment['dialogueacts']:
                    if d['starttime'] not in starttimes:
                        starttimes.append(d['starttime'])
                        preprocessed = preprocess(d['text'])
                        if preprocessed is not None:
                            dialogue.append(preprocessed)
                
                if len(dialogue) > 1:
                    if topic_name not in topic_ids.keys():
                        topic_ids[topic_name] = len(topic_ids)
                    topic_id = topic_ids[topic_name]
                    
                    sections += dialogue
                    labels += [topic_id] * len(dialogue)
                
            if segment['subtopics'] != 'None':
                for subsegment in segment['subtopics']:
                    subtopic_name = topic_name + ': ' + subsegment[self.topic_key]
                    
                    if subsegment['dialogueacts'] != 'None':
                        subdialogue = []
                        for d in subsegment['dialogueacts']:
                            preprocessed = preprocess(d['text'])
                            if preprocessed is not None:
                                subdialogue.append(preprocessed)
                        
                        if len(subdialogue) > 1:
                            if subtopic_name not in topic_ids.keys():
                                topic_ids[subtopic_name] = len(topic_ids)
                            subtopic_id = topic_ids[subtopic_name]
                        
                            sections += subdialogue
                            labels += [subtopic_id] * len(subdialogue)

        return sections, labels, topic_ids

    def _get_sample(self):
        for path in self.textfiles:
            with open(path, 'r') as f:
                segments = json.load(f)
            
            sections, labels, topic_ids = self._get_sections(segments)
            if len(labels) <= 1:
                continue

            boundaries = get_boundaries(labels)
            assert len(sections) == len(labels)
            assert len(sections) == len(boundaries)
            
            targets = [index for index, value in enumerate(list(map(int, boundaries))) if value == 1] + [len(boundaries)]
            
            # Sort topic_ids' keys by values:
            topic_names = sorted(topic_ids, key=topic_ids.get)
            
            yield {'path': str(path),
                   'sections': sections,
                   'labels': labels,
                   'boundaries': boundaries,
                   'split_indices': targets,
                   'topic_names': topic_names}

    def get_generator(self):
        return self._get_sample


def load_dataset_by(cfg):
    if cfg.dataset_type == 'wiki':
        generator = WikiDataset(cfg.input_path).get_generator()
    elif cfg.dataset_type == 'ami':
        generator = AMIDataset(cfg.input_path).get_generator()
    else:
        raise ValueError(f'No such dataset type {cfg.dataset_type} exist!')
    
    ds = Dataset.from_generator(generator)
    if cfg.sample_size is not None:
        ds = Dataset.from_dict(ds[:cfg.sample_size])
    return ds


def calculate_statistics(ds):
    mean_segment_length = 0
    boundaries  = ds['boundaries']
    segment_lengths = [len(b) / (b.count('1') + 1) for b in boundaries]
    mean_segment_length = sum(segment_lengths) / len(segment_lengths)
    return mean_segment_length

# Local dataset:

class SberDataset:
    pass
