# Newspaper Stance Detection: The Gold vs. Silver Debate in the 1890s

This project uses zero-shot NLI stance detection to measure newspaper-level positions on the gold standard vs. free silver debate during the 1890s, using digitized historical newspaper articles from the American Stories dataset.

![Monthly count of gold/silver articles in American newspapers, 1890-1896](figures/timeline_banner.png)

## Background

In the 1890s, the U.S. experienced a major political conflict over monetary policy -- the "Battle of the Standards" -- that split along regional rather than partisan lines. Farmers and miners in the South and West supported the free coinage of silver (to inflate prices and ease debts), while the financial community in the Northeast favored maintaining the gold standard (for currency stability). This project attempts to detect these stances in newspaper coverage using NLP methods. The figure above shows the volume of monetary debate coverage across 212 newspapers, with clear spikes around the Panic of 1893, the repeal of the Sherman Silver Purchase Act, and Bryan's 1896 campaign.

## Method

1. **Data acquisition**: Stream articles from the [American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories) dataset (1890-1896) and filter for articles about the monetary standard debate using domain-specific keywords.
2. **Stance detection**: Apply the [Political DEBATE](https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0) zero-shot NLI model to classify each article as pro-gold, pro-silver, both, or neither.
3. **Aggregation**: Roll up article-level stance scores to the newspaper level, then examine geographic patterns using Library of Congress metadata.

## Project Structure

```
newspaper_stances/
├── notebooks/
│   ├── 01_data_acquisition.ipynb       # Stream & filter American Stories
│   ├── 02_explore_filtered_data.ipynb  # EDA on gold/silver articles
│   ├── 03_stance_detection.ipynb       # Apply Political DEBATE model
│   └── 04_aggregation_analysis.ipynb   # Newspaper-level results + geo
├── src/
│   ├── data_utils.py                   # Streaming/filtering helpers
│   ├── stance_model.py                 # Political DEBATE wrapper
│   └── geo_lookup.py                   # LCCN to geography crosswalk
├── data/
│   ├── american_stories/               # Filtered articles (parquet)
│   ├── lccn_metadata/                  # Geographic crosswalk data
│   └── results/                        # Stance detection outputs
├── requirements.txt
└── README.md
```

## Data Sources

- **American Stories**: Dell Research Harvard, hosted on HuggingFace. A large-scale structured text dataset of historical U.S. newspapers derived from Library of Congress scans.
- **Political DEBATE**: Burnham et al. (2025). A DeBERTa-based NLI model trained on political text for zero-shot and few-shot classification.
- **Library of Congress**: LCCN metadata API for newspaper geographic information.

## Usage

```bash
pip install -r requirements.txt
```

Then run notebooks in order: `01_data_acquisition.ipynb` -> `02_explore_filtered_data.ipynb` -> `03_stance_detection.ipynb` -> `04_aggregation_analysis.ipynb`.

## Known Limitations

This is an early-stage research pipeline. Several limitations should be kept in mind when interpreting results:

- **Northeast undersampled.** Only 11 newspapers across 4 Northeastern states made it through the keyword filter, compared to 41 in the West. The financial-center voices most likely to champion the gold standard are underrepresented, which may partly explain the overall pro-silver skew in the data.
- **Low-count states.** Six states (Maine, Mississippi, Idaho, Nebraska, Oregon, Nevada) are represented by a single newspaper each. Their stance estimates carry high variance and should be treated cautiously.
- **Geographic coverage gaps.** Roughly 81 of 213 LCCNs were resolved via name extraction rather than the Library of Congress API, and some states with active monetary-debate participation (e.g., Ohio, Pennsylvania, Texas) are missing entirely.
- **Pro-gold asymmetry.** The model detects pro-silver rhetoric more readily than pro-gold (mean 0.27 vs. 0.11). Gold standard defenders often used implicit, status-quo framing that may not trigger the zero-shot hypothesis "This text supports the gold standard" as strongly. This is a known challenge with NLI-based stance detection on establishment positions.
- **Flat temporal signal.** Annual averages show little variation from 1890 to 1896, including around the Panic of 1893. This could reflect genuinely entrenched positions, or it could mean annual granularity is too coarse to capture month-level shocks.
- **OCR noise.** The American Stories data is derived from digitized newspaper scans. OCR errors in 1890s print may cause some relevant articles to be missed by keyword filtering or misclassified by the stance model.

## References

- Burnham, M., Kahn, K., Wang, R. Y., & Peng, R. X. (2025). "Political DEBATE: Efficient Zero-Shot and Few-Shot Classifiers for Political Text." *Political Analysis*. https://doi.org/10.1017/pan.2025.10028
- Dell, M., Carlson, J., Bryan, T., Silcock, E., Arora, A., Shen, Z., D'Amico-Wong, L., Le, Q., Querubin, P., & Heldring, L. (2023). "American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers." *NeurIPS 2023 Datasets and Benchmarks*. https://arxiv.org/abs/2308.12477
- Shin, S. (2025). "Measuring Issue Specific Ideal Points from Roll Call Votes." Working paper.
- Frieden, J. (2016). *Currency Politics: The Political Economy of Exchange Rate Policy*.
- Bensel, R. (2000). *The Political Economy of American Industrialization, 1877-1900*.
