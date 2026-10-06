from tqdm.auto import tqdm
tqdm.pandas()

import pandas as pd
import xopen, orjson
import jsonlines
import ast
import functools
import spacy
import os

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('sentencizer')
to_disable = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "textcat"]
label_mapper = {
    'Main': 'Main Event',
    'Main_Consequence': 'Consequences',
    'Cause_General': 'Current Context',
    'Cause_Specific': 'Previous Event',
    'Distant_Evaluation': 'Evaluation',
    'Distant_Expectations_Consequences': 'Expectations',
    'Distant_Historical': 'Historical Event',
    'Distant_Anecdotal': 'Anecdotal Event',
    'NA': 'Error',
}


def read_gzip_nested_jsonl_file(fn):
    dat = []
    with xopen.xopen(fn) as f:
        for line in f:
            dat.append(orjson.loads(line))
    return dat


def format_output(row):
    labels = list(map(lambda x: f'({x[0] + 1}): {x[1]}', enumerate(row["label"])))
    text = list(map(lambda x: f'({x[0] + 1}): {x[1]}', enumerate(row["sent"])))
    label_text = ', '.join(labels).strip()
    text = ' '.join(text).strip()
    header = row.fillna('')["snippet_to_use"].strip()
    header = header if (header.endswith('.')) else (header + '.')

    return f'''{header} <labels> {label_text} <text> {text}'''


def get_sents(sent_list):
    pipe = nlp.pipe(sent_list, disable=to_disable)
    pipe = tqdm(pipe, total=len(sent_list))
    sentences = []
    for doc in pipe:
        sents = list(map(str, doc.sents))
        sentences.append(sents)
    return sentences


def robust_ast(x):
    try:
        return ast.literal_eval(x)
    except:
        return


def merge_with_metadata(source_sent_list):
    metadata_df = pd.read_csv('../data/2021-2023__current-articles/nytimes-article-metadata.csv.gz')
    metadata_df = metadata_df.assign(headline=lambda df: df['headline'].apply(robust_ast).str.get('main'))
    metadata_df['headline_sents'] = get_sents(metadata_df['headline'].fillna(''))
    metadata_df['snippet_sents'] = get_sents(metadata_df['snippet'].fillna(''))
    metadata_df['snippet_to_use'] = metadata_df.apply(
        lambda x: x['snippet_sents'][0] if len(x['snippet_sents']) == 1 else x['headline'], axis=1
    )

    return (
        source_sent_list
            .reset_index()
            .assign(index=lambda df: df['doc_idx'].str.split('\d{14}/', regex=True).str.get(1).str.strip())
            .merge(metadata_df[['web_url', 'snippet_to_use']], left_on='index', right_on='web_url')
            .drop(['index', 'web_url',], axis=1)
    )


# get source labels by MPI
def convert_sentence_labels_to_source_labels_by_mpi(sentence_labeled_df, label_col, attribution_df):
    type_likelihoods_by_source = (
        attribution_df
            .merge(sentence_labeled_df, on=['doc_idx', 'sent_idx'])
            .groupby(['doc_idx', 'attribution'])[label_col]
            .value_counts()
            .unstack().fillna(0)
            .pipe(lambda df: df.divide(df.sum(axis=1), axis=0))
    )

    overall_type_likelihood = sentence_labeled_df[label_col].value_counts().pipe(lambda s: s / s.sum())
    return (type_likelihoods_by_source / overall_type_likelihood).idxmax(axis=1)


def get_attribution_and_discourse_df():
    attribution_fn = '../data/nyt-2021-2023/slim-nyt-2021-2023-attribution-outfile.jsonl.gz'
    discourse_fn = '../data/nyt-2021-2023/slim-nyt-2021-2023-baseline-metrics___discourse.jsonl.gz'
    attribution_dat = read_gzip_nested_jsonl_file(attribution_fn)
    discourse_dat = read_gzip_nested_jsonl_file(discourse_fn)
    attribution_df = pd.concat(list(map(pd.DataFrame, attribution_dat)))
    discourse_df = pd.concat(list(map(pd.DataFrame, discourse_dat)))
    discourse_df['discourse_type'] = discourse_df['discourse_type'].map(label_mapper)
    # remove NA sentences
    attribution_df = (
        attribution_df
             .merge(discourse_df, on=['doc_idx', 'sent_idx'])
             .loc[lambda df: ~df['discourse_type'].isin(['NA', 'Error'])]
             .drop(columns=['discourse_type'])
             .loc[lambda df: df['is_quote'] == True]
             .loc[lambda df: ~df['attribution'].isin(['None', 'oom error'])]
             .loc[lambda df: df['attribution'].str.split().str.len() < 8]
             .loc[lambda df: ~df['quote_type'].isin([
                'No Quote',
                # 'Background/Narrative',
                'oom error',
             ])]
    )
    return attribution_df, discourse_df


def get_source_level_labels_from_sentence_level_datatypes(
        source_label_type, discourse_df=None, attribution_df=None
):
    ## read in data
    if source_label_type == 'argumentation':
        argumentation_fn = '../data/nyt-2021-2023/slim-nyt-2021-2023-baseline-metrics___argumentation.jsonl.gz'
        argumentation_dat = read_gzip_nested_jsonl_file(argumentation_fn)
        label_df = pd.concat(list(map(pd.DataFrame, argumentation_dat)))

    elif source_label_type == 'nli':
        nli_fn = '../data/nyt-2021-2023/slim-nyt-2021-2023-baseline-metrics___nli.jsonl.gz'
        nli_dat = read_gzip_nested_jsonl_file(nli_fn)
        label_df = pd.concat(list(map(pd.DataFrame, nli_dat)))

    elif source_label_type == 'stance':
        stance_fn = '../data/2021-2023__current-articles/slim-nyt-2021-2023-baseline-metrics___stance.csv.gz'
        label_df = pd.read_csv(stance_fn, index_col=0)

    elif source_label_type == 'discourse':
        label_df = discourse_df

    elif source_label_type == 'quote_type':
        label_df = attribution_df[['doc_idx', 'sent_idx', 'quote_type']]

    ## aggregate labels
    get_max_source_mpi = functools.partial(
        convert_sentence_labels_to_source_labels_by_mpi,
        attribution_df=attribution_df
    )
    if source_label_type in ['argumentation', 'nli', 'discourse', 'quote']:
        col_name = type_to_label_mapper[source_label_type]
        source_labels = get_max_source_mpi(label_df, col_name)

    elif source_label_type == 'stance':
        stance_likelihoods = label_df['y_pred'].value_counts().pipe(lambda s: s / s.sum())
        source_stance_likelihoods = (
            label_df
                .groupby(['doc_id', 'source'])['y_pred']
                .value_counts().unstack()
                .fillna(0).pipe(lambda df: df.divide(df.sum(axis=1), axis=0))
        )
        source_labels = (source_stance_likelihoods / stance_likelihoods).idxmax(axis=1)

    return source_labels


type_to_label_mapper = {
    'argumentation': 'argumentation_type',
    'nli': 'nli_label',
    'discourse': 'discourse_type',
    'stance': 'stance',
    'quote_type': 'quote_type',
}
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source-label-types', nargs='+', default=['argumentation', 'nli', 'discourse', 'stance', 'quote_type'])
    p.add_argument('--num-labels', type=int, default=5)
    p.add_argument('--num-sentences-per-source', type=int, default=4)
    p.add_argument('--num-sources-per-story', type=int, default=15)
    p.add_argument('--output-dir', type=str, default='tmp')

    args = p.parse_args()

    attribution_df, discourse_df = get_attribution_and_discourse_df()
    source_sent_list_w_metadata = (
        attribution_df
             .groupby(['doc_idx', 'attribution'])['sent']
             .aggregate(list)
             .apply(lambda x: x[:4])
             .str.join(' ')
             .pipe(merge_with_metadata)
    )

    for source_label_type in args.source_label_types:
        if source_label_type == 'random':
            pass
        if source_label_type in ['argumentation', 'nli', 'discourse', 'stance', 'quote_type']:
            source_labels = get_source_level_labels_from_sentence_level_datatypes(
                source_label_type, discourse_df=discourse_df, attribution_df=attribution_df
            )
        if source_label_type in ['topic-model', 'affiliation', 'role', 'source-type']:
            source_labels = pd.read_csv(f'../data/nyt-2021-2023/slim-nyt-2021-2023-baseline-metrics___{source_label_type}.csv.gz', index_col=0)['y_pred']


        fname = os.path.join(args.output_dir, source_label_type.replace('_', '-'))
        (
            source_sent_list_w_metadata
             .merge(
                source_labels.to_frame('label').reset_index(),
                on=['doc_idx', 'attribution'],
             )
             .reset_index().groupby('doc_idx')[['sent', 'label']]
             .aggregate(list)
             .map(lambda x: x[:args.num_sources_per_story])
             ##
             .apply(format_output, axis=1)
             .to_frame('text')
             .to_csv(
                 '../modeling/gpt_finetuning/data/argumentation-formatted-data.csv.gz',
                 compression='gzip'
             )
        )