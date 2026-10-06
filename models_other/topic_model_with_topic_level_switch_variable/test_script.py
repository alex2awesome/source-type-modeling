import json, os
from tqdm.auto import tqdm

seen_labels = {}
with open('input_data/doc_source.json') as f:
    doc_strs = f.read().split('\n')
    docs = []
    for idx, doc_str in enumerate(doc_strs):
        if doc_str:
            doc = json.loads(doc_str)
            input_doc = {}
            input_doc['has_source_labels'] = len(doc.get('source_labels', [])) > 0
            doc_label = doc.get('doc_label')
            input_doc['has_doc_label'] = doc_label is not None
            if input_doc['has_doc_label']:
                if doc_label not in seen_labels:
                    seen_labels[doc_label] = len(seen_labels)
                input_doc['doc_label'] = seen_labels[doc_label]
            input_doc['source_vecs'] = list(doc.get('source_vecs', {}).values())
            input_doc['doc_vec'] = doc['doc_vec']
            docs.append(input_doc)

vocab_fp = os.path.join('input_data', 'vocab.txt')
vocab = open(vocab_fp).read().split('\n')

import sampler_cy
s2 = sampler_cy.BOW_Source_GibbsSampler(
    docs=docs,
    vocab=vocab,
    num_doctypes=len(seen_labels),
    num_topics=10,

)
s2.initialize()
for i in tqdm(range(100)):
    s2.sample_pass()
s2.pythonize_vars()
s2.save_state('output_data')
# print(s2.source_to_type_list)
# print(s2.sourcetype_by_wordtopic__wordtopic_counts)