import os, json
import glob
import dill as pickle
import re
from tqdm.auto import tqdm


def process_data(docs, use_cython=False):
    output_docs = []
    seen_doc_labels = {}
    for doc in docs:
        processed_doc = {}
        processed_doc['has_source_labels'] = len(doc.get('source_labels', [])) > 0
        doc_label = doc.get('doc_label')
        processed_doc['has_doc_label'] = doc_label is not None
        if processed_doc['has_doc_label']:
            if doc_label not in seen_doc_labels:
                seen_doc_labels[doc_label] = len(seen_doc_labels)
            processed_doc['doc_label'] = seen_doc_labels[doc_label]
        if use_cython:
            processed_doc['source_vecs'] = list(doc.get('source_vecs', {}).values())
        else:
            processed_doc['source_vecs'] = doc.get('source_vecs', {})
            processed_doc['source_map'] = doc.get('source_map', {})
        processed_doc['doc_vec'] = doc['doc_vec']
        output_docs.append(processed_doc)
    return output_docs, len(seen_doc_labels)


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    # model params
    p.add_argument('--input-dir', type=str, help="input directory.")
    p.add_argument('--output-dir', type=str, help="output directory.")
    p.add_argument('--num-topics', type=int, help="num topics.")
    p.add_argument('--num-docs', type=int, help="num docs.")
    p.add_argument('--num-source-types', type=int, help="num personas.")
    p.add_argument('--num-iter', type=int, default=10, help="num iterations.")
    p.add_argument('--use-cached', action='store_true', help='use intermediate cached file.')
    p.add_argument('--use-cython', action='store_true', help='use cython-based sampler.')
    args = p.parse_args()

    here = os.path.dirname(__file__)
    input_documents_fp = os.path.join(here, args.input_dir, 'doc_source.json')
    output_dir = os.path.join(here, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_documents_fp) as f:
        doc_strs = f.read().split('\n')
        docs = []
        for idx, doc_str in enumerate(doc_strs):
            if doc_str:
                doc = json.loads(doc_str)
                docs.append(doc)
            if (args.num_docs is not None) and (idx > args.num_docs):
                break

    vocab_fp = os.path.join(args.input_dir, 'vocab.txt')
    vocab = open(vocab_fp).read().split('\n')

    num_doc_types = 10
    if args.use_cython:
        from sampler_cy import BOW_Source_GibbsSampler as sampler_class
    else:
        from sampler import BOW_Source_GibbsSampler as sampler_class

    docs, num_doc_types = process_data(docs)

    use_source_labels = False
    if use_source_labels:
        num_source_types = len(open(os.path.join(args.input_dir, 'roles.txt')).read().split('\n'))
        sampler = sampler_class(
            docs=docs,
            vocab=vocab,
            num_source_types=num_source_types,
            num_doctypes=num_doc_types,
        )
    else:
        sampler = sampler_class(
            docs=docs,
            vocab=vocab,
            num_doctypes=num_doc_types,
            use_doc_labels=False
        )

    ##
    cached_files = glob.glob(os.path.join(output_dir, 'trained-sampled-iter*'))
    if not args.use_cached or (len(cached_files) == 0):
        prev_iter = 0
        sampler.initialize()
    else:
        print('loading...')
        max_file = max(cached_files, key=lambda x: int(re.findall('iter-(\d+)', x)[0]))
        sampler = pickle.load(open(max_file, 'rb'))
        prev_iter = int(re.findall('iter-(\d+)', max_file)[0])

    for i in tqdm(range(args.num_iter), total=args.num_iter):
        if (i % 20 == 0) and args.use_cached:
            pickle.dump(sampler, open(os.path.join(output_dir, 'trained-sampled-iter-%d.pkl' % (i + prev_iter)), 'wb'))
        sampler.sample_pass()

    ## done
    pickle.dump(sampler, open(os.path.join(output_dir, 'trained-sampled-iter-%d.pkl' % i), 'wb'))


'''
python sampler_runner.py \
    --input-dir input_data \
    --output-dir output_data_no_doc_labels \
    --num-topics 10 \
    --num-source-types 10 \
    --use-cached \
    --num-iter 500
'''