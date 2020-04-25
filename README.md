# Influence function analysis

The main environment requirements for this project are `python 3.6`, `pytorch 1.2.0`, and `pytorch-pretrained-bert 0.6.1`.

`run_tagger.sh`: finetune a BERT model for sentiment analysis (SA) and natural language inference (NLI).

`run_influence.sh`: generate influence function results in the paper, including influence score calculation, leave-one-out (LOO) training, and token removals.

`SA_analysis.ipynb`: see influential examples in SA (Figure 1 in the paper).

`NLI_analysis.ipynb`: see influential examples in NLI (Table 5 in the paper).

`LOO.ipynb`: see LOO training results for both SA and NLI (Table 1 and 2 in the paper).

`SA_sal_if_direct_consistency.ipynb`, `NLI_sal_if_direct_consistency.ipynb`: Figure 2 and 3 in the paper.

`SA_mask_token_analysis.ipynb`, `NLI_mask_token_analysis.ipynb`: Table 3 and 4 in the paper.

`NLI_heuristic_analysis.ipynb`: Section 5.1 in the paper.

If you have any questions, please email Xiaochuang Han at xiaochuang.han@gmail.com. Thank you!
