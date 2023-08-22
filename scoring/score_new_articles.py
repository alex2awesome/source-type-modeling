import sys
import os
from transformers import AutoTokenizer, AutoConfig
from unidecode import unidecode
import re
import jsonlines
import pandas as pd
import torch
from tqdm.auto import tqdm
from more_itertools import flatten
import json
from itertools import groupby

CLEANR = re.compile('<.*?>')
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, '../modeling/quote-type-modeling/src/'))
sys.path.insert(0, os.path.join(here, '../modeling/source-type-modeling/src/'))
from sentence_model import SentenceClassificationModel
from sentence_model import TokenizedDataset
from batch_sequence_util import cached_label_mapper as label_mapper
from batch_sequence_modeling import DocumentClassificationModel
from batch_sequence_modeling import TokenizedDataset as DocumentTokenizedDataset
from batch_sequence_modeling import collate_fn as document_collate_fn

MAX_SENTENCE_LEN_PER_DOC = 110

class QADatasetWrapper():
    def __init__(self, qa_dataset, tokenizer, collator, device=None):
        self.qa_dataset = qa_dataset
        self.collator = collator
        self.device = device or get_device()
        self.tokenizer = tokenizer

    def process_one_doc(self, doc):
        return self.qa_dataset.process_one_doc(doc)

    def prepare_sent(self, input_packet):
        output = {}
        cols = ['input_ids', 'token_type_ids']
        for col in cols:
            if col in input_packet:
                output[col] = (
                    torch.tensor(input_packet[col])
                        .unsqueeze(dim=0)
                        .to(self.device)
                )
        return output

    def process_output(self, attribution, input_packet):
        start_token, end_token = list(map(lambda x: x.argmax(), attribution))
        start_token, end_token = min(start_token, end_token), max(start_token, end_token)
        span = input_packet['input_ids'][0, start_token: end_token + 1].detach().cpu().numpy()
        return self.tokenizer.decode(span)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def get_attribution_model_and_dataset(args):
    #
    from qa_model import QAModel, QATokenizedDataset, collate_fn

    attribution_config = AutoConfig.from_pretrained(args.attribution_model, cache_dir=args.cache_dir)
    attribution_tokenizer = AutoTokenizer.from_pretrained(
        args.attribution_tokenizer or args.attribution_model, cache_dir=args.cache_dir
    )
    attribution_model = QAModel.from_pretrained(args.attribution_model, cache_dir=args.cache_dir)
    attribution_model = attribution_model.to(device)
    attribution_dataset_core = QATokenizedDataset(hf_tokenizer=attribution_tokenizer)
    attribution_dataset = QADatasetWrapper(
        qa_dataset=attribution_dataset_core, collator=collate_fn, tokenizer=attribution_tokenizer
    )
    return attribution_dataset, attribution_model


def filter_cached_detection_data(args):
    data_with_detection = list(jsonlines.open(args.detection_outfile))
    if args.already_run_ids:
        ran_ids = json.load(open(args.already_run_ids))
        data_with_detection = list(filter(lambda x: not x[0]['doc_idx'] in ran_ids, data_with_detection))
    if args.to_run_ids:
        to_run_ids = json.load(open(args.to_run_ids))
        data_with_detection = list(filter(lambda x: x[0]['doc_idx'] in to_run_ids, data_with_detection))
    if args.start_idx is not None:
        data_with_detection = data_with_detection[args.start_idx:]
    if args.n_docs is not None:
        data_with_detection = data_with_detection[:args.n_docs]
    return data_with_detection


spacy_model = None
def get_spacy_model():
    global spacy_model
    if spacy_model is None:
        import spacy
        spacy_model = spacy.load('en_core_web_lg')
    return spacy_model


def process_spacy_doc_for_sents(doc):
    sents = list(map(str, doc.sents))
    sents = list(flatten(map(lambda x: x.split('\n'), sents)))
    sents = list(map(lambda x: unidecode(x).strip(), sents))
    sents = list(filter(lambda x: x != '', sents))
    sents = list(map(lambda x: re.sub(r' +', ' ', x), sents))
    return sents

def sentencize_doc(body):
    spacy_model = get_spacy_model()
    body = cleanhtml(body)
    doc = spacy_model(body)
    sents = process_spacy_doc_for_sents(doc)
    sents = sents[:MAX_SENTENCE_LEN_PER_DOC]
    return sents

def sentencize_col(text_col):
    spacy_model = get_spacy_model()
    spacy_model.add_pipe('sentencizer')
    doc_sentences = []
    pipeline_process= spacy_model.pipe(text_col, disable=[
        "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "textcat"
    ])
    for doc in tqdm(pipeline_process, total=len(text_col)):
        sents = process_spacy_doc_for_sents(doc)
        sents = sents[:MAX_SENTENCE_LEN_PER_DOC]
        doc_sentences.append(sents)

    return doc_sentences

def make_attribution_file_name(args):
    fn, f_end = args.attribution_outfile.split('.')
    s, e = args.start_idx or 0, (args.start_idx or 0) + (args.n_docs or 0)
    args.attribution_outfile = fn + f'__{s}-{e}__' + '.' + f_end
    return args


def add_source_attribute(all_docs_source_groups, tokenizer_name, model_name, attribute_group_name):
    config = AutoConfig.from_pretrained(model_name)
    max_length = config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = DocumentTokenizedDataset(
        label_mapper=label_mapper[attribute_group_name],
        tokenizer=tokenizer,
        do_score=True,
        max_length=max_length - 20
    )
    model = (
        DocumentClassificationModel
        .from_pretrained(model_name)
        .to(device)
    )
    output_source_groups = []
    for doc_source_groups in all_docs_source_groups:
        doc_source_group_sans_none = list(filter(lambda x: x['name'] != None, doc_source_groups))
        source_texts = list(map(lambda x: x['text'], doc_source_group_sans_none))
        #
        if len(source_texts) > 0:
            d = list(map(dataset.process_one_doc, source_texts))
            input = document_collate_fn(d, padding_value=tokenizer.pad_token_id)
            input = {k: v.to(device) for k, v in input.items()}
            logits = model(**input)[0]
            preds = dataset.transform_logits_to_labels(logits)

            # map preds into our packets
            name_to_pred = dict(zip(list(map(lambda x: x['name'], doc_source_group_sans_none)), preds))
        else:
            name_to_pred = {}

        # append
        doc_source_output = []
        for source_packet in doc_source_groups:
            source_packet[attribute_group_name] = name_to_pred.get(source_packet['name'], None)
            doc_source_output.append(source_packet)
        output_source_groups.append(doc_source_output)

    return output_source_groups

# python score_new_articles.py \
#   --detection_model alex2awesome/quote-detection__roberta-base-sentence \
#   --detection_tokenizer roberta-base \
#   --detection_outfile data_with_detection.jsonl \
#   --attribution_tokenizer google/bigbird-roberta-base \
#   --attribution_model alex2awesome/quote-attribution__qa-model \
#   --attribution_outfile test_new.jsonl \
#   --dataset_name data_to_score_longer.jsonl \
#   --do_attribution \
#   --do_detection \
#   --n_docs 10 \
#   --attribution_outfile temp.jsonl

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--detection-model', type=str, )
    parser.add_argument('--detection-config', type=str, default=None)
    parser.add_argument('--detection-tokenizer', default=None, type=str, )
    parser.add_argument('--is-quote-cutoff', default=.5, type=float)
    parser.add_argument('--detection-outfile', default=None, type=str)
    parser.add_argument('--do-detection', action='store_true')
    #
    parser.add_argument('--attribution-model', type=str )
    parser.add_argument('--attribution-tokenizer', default='gpt', type=str, )
    parser.add_argument('--attribution-outfile', type=str)
    parser.add_argument('--do-attribution', action='store_true')
    #
    parser.add_argument('--do-quote-type-classification', action='store_true')
    parser.add_argument('--quote-type-model', type=str, )
    parser.add_argument('--quote-type-tokenizer', default=None, type=str, )

    parser.add_argument('--do-source-type-classification', action='store_true')
    parser.add_argument('--source-type-model', type=str, )
    parser.add_argument('--source-type-tokenizer', default=None, type=str, )

    parser.add_argument('--do-affiliation-classification', action='store_true')
    parser.add_argument('--affiliation-model', type=str, )
    parser.add_argument('--affiliation-tokenizer', default=None, type=str, )

    parser.add_argument('--do-role-classification', action='store_true')
    parser.add_argument('--role-model', type=str, )
    parser.add_argument('--role-tokenizer', default=None, type=str, )

    # dataset args
    parser.add_argument('--dataset-name', type=str, )
    parser.add_argument('--body-col-name', type=str, default='body')
    parser.add_argument('--id-col-name', type=str, default='suid')
    parser.add_argument('--start-idx', default=None, type=int)
    parser.add_argument('--n-docs', default=None, type=int)
    parser.add_argument('--to-run-ids', default=None, type=str)
    parser.add_argument('--already-run-ids', default=None, type=str)
    #
    parser.add_argument('--source-attribute-outfile', default=None, type=str)
    parser.add_argument('--platform', default='local', type=str)
    args = parser.parse_args()

    # load in dataset
    data = pd.read_csv(args.dataset_name).loc[lambda df: df[args.body_col_name].notnull()]
    if args.start_idx is not None:
        data = data.iloc[args.start_idx:]

    if args.n_docs is not None:
        data = data.iloc[:args.n_docs]

    if 'sentences' not in data.columns:
        print('sentencizing...')
        data['sentences'] = data[args.body_col_name].pipe(sentencize_col)

    device = get_device()
    # Load the models
    args.cache_dir = None

    # detection
    if args.do_detection:
        print('running detection...')
        detection_tokenizer = AutoTokenizer.from_pretrained(args.detection_tokenizer or args.detection_model, cache_dir=args.cache_dir)
        detection_config = AutoConfig.from_pretrained(args.detection_config or args.detection_model, cache_dir=args.cache_dir)
        detection_model = SentenceClassificationModel.from_pretrained(args.detection_model, config=detection_config, cache_dir=args.cache_dir)
        detection_dataset = TokenizedDataset(
            tokenizer=detection_tokenizer, do_score=True)

        # perform detection
        detection_model.eval()
        detection_model = detection_model.to(device)

        # stream the data to a file
        data_with_detection = []
        for idx, doc in tqdm(data.iterrows(), total=len(data)):
            sentences = doc['sentences']
            input_ids, attention_mask, _ = detection_dataset.process_one_doc(sentences)
            if input_ids is not None:
                processed_datum = {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }

                # perform quote detection score
                try:
                    scores = detection_model.get_proba(**processed_datum)
                    scores = scores.cpu().detach().numpy().flatten()
                    datum = []
                    for sent_idx, sent in enumerate(sentences):
                        output_packet = {
                            'is_quote': (float(scores[sent_idx]) > args.is_quote_cutoff),
                            'sent': sent,
                            'sent_idx': sent_idx,
                            'doc_idx': doc[args.id_col_name],
                        }
                        datum.append(output_packet)
                    data_with_detection.append(datum)
                except:
                    print('detection error')

        if args.detection_outfile is not None:
            with open(args.detection_outfile, 'w') as f:
                jsonlines.Writer(f).write_all(data_with_detection)

        del detection_model
        del detection_tokenizer

    # quote type classification
    if args.do_quote_type_classification:
        print('running quote_type_classification...')
        quote_type_tokenizer = AutoTokenizer.from_pretrained(args.detection_tokenizer or args.detection_model, cache_dir=args.cache_dir)
        quote_type_config = AutoConfig.from_pretrained(args.quote_type_model, cache_dir=args.cache_dir)
        quote_type_model = (
            SentenceClassificationModel
                .from_pretrained(args.quote_type_model, config=quote_type_config, cache_dir=args.cache_dir)
                .to(device)
        )
        quote_type_label_mapper = label_mapper['quote-type']
        quote_type_dataset = TokenizedDataset(
            tokenizer=quote_type_tokenizer, do_score=True, label_mapper=quote_type_label_mapper
        )
        for idx, doc in enumerate(tqdm(data_with_detection)):
            sentences = list(map(lambda x: x['sent'], doc))
            d = quote_type_dataset.process_one_doc(sentences)
            if (d[0] is not None) and (d[1] is not None):
                d = {
                    'input_ids': d[0],
                    'attention_mask': d[1]
                }
                d = {k: v.to(device) for k, v in d.items()}
                _, logits = quote_type_model.process_one_doc(**d)
                preds = quote_type_dataset.transform_logits_to_labels(logits, num_docs=len(d['input_ids']))
                for sent_idx, sent in enumerate(sentences):
                    data_with_detection[idx][sent_idx]['quote_type'] = preds[sent_idx]
        del quote_type_model
        del quote_type_tokenizer

    if args.do_attribution:
        print('running attribution...')
        args = make_attribution_file_name(args)
        attribution_dataset, attribution_model = get_attribution_model_and_dataset(args)

        if not args.do_detection:
            data_with_detection = filter_cached_detection_data(args)

        with open(args.attribution_outfile, 'w') as f:
            writer = jsonlines.Writer(f)

            # stream the data to a file
            for doc_idx, datum_for_attribution in enumerate(tqdm(data_with_detection)):
                # perform attribution
                final_output = []
                data_for_scoring = attribution_dataset.process_one_doc(datum_for_attribution)
                for sent_idx, (packet, datum) in enumerate(zip(datum_for_attribution, data_for_scoring)):
                    if datum is not None and packet['is_quote']:
                        datum = attribution_dataset.prepare_sent(datum)
                        attribution = attribution_model(**datum)
                        attribution = attribution_dataset.process_output(attribution, datum)
                    else:
                        attribution = None
                    # add attribution to the packet
                    data_with_detection[doc_idx][sent_idx]['sent'] = data_with_detection[doc_idx][sent_idx]['sent'].replace('journalist passive-voice ', '')
                    data_with_detection[doc_idx][sent_idx]['attribution'] = attribution

                # stream to disk (check!!)
                writer.write(data_with_detection[doc_idx])

        del attribution_model
        del attribution_dataset

    all_source_groups = []
    for doc_idx, datum in enumerate(data_with_detection):
        datum = sorted(datum, key=lambda x: str(x['attribution']))
        source_groups = []
        for name, sent_iter in groupby(datum, key=lambda x: x['attribution']):
            source_groups.append({
                'name': name,
                'doc_id': datum[0]['doc_idx'],
                'text': ' '.join(list(map(lambda x: x['sent'], sent_iter)))
            })
        all_source_groups.append(source_groups)

    if args.do_source_type_classification:
        all_source_groups = add_source_attribute(
            all_source_groups, args.source_type_tokenizer, args.source_type_model, 'source-type')

    if args.do_affiliation_classification:
        if args.affiliation_tokenizer is None:
            args.affiliation_tokenizer = args.source_type_tokenizer

        all_source_groups = add_source_attribute(
            all_source_groups, args.affiliation_tokenizer, args.affiliation_model, 'affiliation')

    if args.do_role_classification:
        if args.role_tokenizer is None:
            args.role_tokenizer = args.affiliation_tokenizer

        all_source_groups = add_source_attribute(
            all_source_groups, args.role_tokenizer, args.role_model, 'role')

    with open(args.source_attribute_outfile, 'w') as f:
        jsonlines.Writer(f).write_all(all_source_groups)





