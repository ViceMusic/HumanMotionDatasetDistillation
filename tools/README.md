# Tools

This folder stores reusable helper code for data processing and visualization.

Converted data should generally use `.npz` format. A typical file may contain:

```text
motions      # [N, T, J, 3]
joint_names  # [J]
edges        # [E, 2]
fps          # int
```

