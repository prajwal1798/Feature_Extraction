# Archive (not imported by the active code)

Earlier scripts, kept for reference. They import the module names of an earlier layout (`src.model`, `src.runtime`, ...) and do not run against the present code without porting.

- `pod/`: POD-DeepONet: basis (method of snapshots), model, preparation, selection, validation, rank check, completion job. Not part of the present comparison.
- `depth_study/`: the DeepONet depth-4/6 study. Superseded by the DeepONet tuning grids (depths 2-10, tanh and GELU).
- `paper_figures/`: the earlier report-figure suite (final_figures, paper_plots, plot_comparison, ...). It requires POD results; the DeepONet/FNO figures are now produced by `src/analysis/report.py`.
