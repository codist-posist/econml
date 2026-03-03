# Tables And Figures Coverage (SSRN 5005047)

This repository keeps legacy notebooks and adds a paper-aligned layer.

## Tables

- Table 1: [101_paper_tables_1_4.ipynb](/c:/Users/User/Desktop/econml/notebooks/101_paper_tables_1_4.ipynb) (`table1_calibration.csv`)
- Table 2: [101_paper_tables_1_4.ipynb](/c:/Users/User/Desktop/econml/notebooks/101_paper_tables_1_4.ipynb) (`table2_sim_conditional.csv` / `table2_fixed_point.csv`)
- Table 3: [101_paper_tables_1_4.ipynb](/c:/Users/User/Desktop/econml/notebooks/101_paper_tables_1_4.ipynb) (`table3_sim_conditional.csv` / `table3_fixed_point.csv`)
- Table 4: [101_paper_tables_1_4.ipynb](/c:/Users/User/Desktop/econml/notebooks/101_paper_tables_1_4.ipynb) (`table4_sim_conditional.csv` / `table4_fixed_point.csv`)

## Figures

- Figure 1: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 2: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 4: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 7: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 8: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 9: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 10: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Figure 12: [115_paper_figures_2_14.ipynb](/c:/Users/User/Desktop/econml/notebooks/115_paper_figures_2_14.ipynb)
- Supplementary (not core paper figures): regime-change Taylor plot, temporary-shock comparison plot, and ZLB plots.

## Notes

- Existing notebooks (`91/92/93/94/96/99/100`) are preserved.
- Paper table construction supports two SSS sources:
  - `sim_conditional` (author-style long-simulation conditional means),
  - `fixed_point` (policy fixed-point regime values).
- `115_paper_figures_2_14.ipynb` now encodes article-style scenarios for core figures and keeps extras as supplementary:
  - regime-switch figures start at normal SSS and set temporary innovations to zero,
  - cost-push IRFs use an initial `xi` jump and zero later innovations (AR(1) propagation),
  - Figure 9 uses `eta_bar=0` and fixed normal regime,
  - Figure 12 uses up to 10,000 simulated quarters.
- If ZLB runs are not available yet, Table 4 and Figures 10/11/14 cannot be fully populated.
