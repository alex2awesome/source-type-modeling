import pandas as pd
import spacy
from tqdm.auto import tqdm
from accelerate import PartialState
import os


def get_lemmas(text=None, doc=None):
    if doc is None:
        doc = nlp(text)
    lemmas = list(map(lambda x: x.lemma_, doc))
    return ' '.join(lemmas)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--distribute', action='store_true')
    args = parser.parse_args()

    if args.distribute:
        distributed_state = PartialState()

    all_attr_dfs = pd.read_csv(args.input_file)
    nlp = spacy.load('en_core_web_lg')

    if args.distribute:
        rows_to_process = all_attr_dfs.to_dict(orient='records')
        with distributed_state.split_between_processes(rows_to_process) as data_per_process:
            all_attr_dfs = pd.DataFrame(data_per_process)
            process_num = distributed_state.process_index
            process_name = f'Process: {process_num}.'
            old_outfile_name, old_outfile_ext = os.path.splitext(args.output_file)
            new_outfile = f'{old_outfile_name}__process-{process_num}{old_outfile_ext}'
            args.output_file = new_outfile

    sents = all_attr_dfs['sent']
    lemmas = []
    for doc in tqdm(nlp.pipe(sents), total=len(sents)):
        lemmas.append(get_lemmas(doc=doc))

    all_attr_dfs['lemmas'] = lemmas
    all_attr_dfs.to_csv(args.output_file,index=False)


"""
accelerate launch --num_processes 4 lemmatize.py \
    --input-file tmp/attr-df.csv  \
    --output-file tmp/attr-df-w-lemmas.csv \
    --distribute
"""
