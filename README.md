# source-type-modeling

Code for modeling the informational sources that news articles quote or cite. We ask why a story includes the particular mix of sources it does. Starting from attribution lines (who is quoted and what they said), we describe each source along several schemas: stance, argumentation, discourse role, textual entailment, and three we introduce (retrieval channel, social affiliation, and role in the story). We then treat the schema a journalist used as a latent variable and ask which schema best explains the sources in each document. The scoring scripts run the full annotation pipeline over new articles (quote detection, attribution, quote type, source type, affiliation, role) using classifiers published on the Hugging Face Hub under `alex2awesome/`.

## Related paper

*Explaining Mixtures of Sources in News Articles* (EMNLP 2024). Earlier drafts in `presentation/`: *Learning Mixture of Sources for Retrieval* (NAACL 2024 submission) and *Identifying Mixtures of Sources in News Article as a Document-Planning Process* (ACL 2024 submission). The paper releases *NewsSources*, schema annotations for about 4M articles.

## Layout

- `scoring/` -- `score_new_articles.py` runs detection, attribution and the source classifiers over a CSV of articles; `score_by_baseline_metrics.py` adds stance, argumentation, discourse and NLI labels; `run-scoring.sh` is an example; `qa_model.py` is the attribution model.
- `source-retrieval/` -- FAISS / `retriv` indexes for the source-recommendation experiments.
- `notebooks/` -- dated notebooks (2023-2024): NYT Annotated Corpus processing, classifier training, topic models over source types, GPT fine-tuning, paper figures.
- `presentation/` -- LaTeX and figures for the three paper versions.

## How to run

`pip install -r requirements.txt`, plus spaCy `en_core_web_lg` and NLTK stopwords. Example:

```
python scoring/score_new_articles.py --dataset-name articles.csv --id-col-name url --body-col-name text \
  --do-detection --do-attribution --do-source-type-classification \
  --do-affiliation-classification --do-role-classification --source-attribute-outfile out.jsonl
```

`scoring/` imports model code from `../modeling/quote-type-modeling/src` and `../modeling/source-type-modeling/src`, which are not in this repository. GPT steps read an OpenAI key from a file in the home directory.

## Data

Not included (`data/`, `*.csv`, `*.json`, `*.jsonl` are gitignored). Inputs are news articles with attribution-line annotations, mainly the NYT Annotated Corpus (LDC) plus newer NYT and local-news scrapes. `notebooks/cache/` holds large untracked intermediates.

## Status

Scripts last changed January 2024; notebooks through August 2024; camera-ready LaTeX November 2024.
