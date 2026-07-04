# Result Artifacts

This directory stores generated-query CSV files and evaluation exports.

The current evaluation CLI writes one folder per generated file:

```text
results/
  <generated-file-name>/
    stats.csv
    metrics.csv
```

Existing `test_data_*` files are preserved reference artifacts from the thesis workflow. They should not be interpreted as a fresh model benchmark unless regenerated with the documented inference and evaluation commands.
