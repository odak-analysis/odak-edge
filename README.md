# odak-edge

The in-store part of ODAK (Observe, Detect, Analyze, Know). It runs on a computer inside the store, reads the store's existing security cameras and turns the video into anonymous, aggregated numbers. Video never leaves the store. This repository also holds the setup tool that is used to install and configure a store.

This repository currently contains the structure only. Code, dependencies and the Python version are added with the first development tasks.

## Structure

```
odak-edge/
├── analysis/      Analysis program: camera video in, aggregated numbers out
├── setup_tool/    Setup tool: installs and configures the edge module for a store
├── shared/        Code used by both the analysis program and the setup tool
├── tests/         Tests
├── .gitattributes
├── .gitignore
└── README.md
```

Each folder has a short `README.md` that says what belongs in it.

## Related repositories

| Repository | Content |
|---|---|
| [`odak-server`](https://github.com/odak-analysis/odak-server) | Central server |
| [`odak-dashboard`](https://github.com/odak-analysis/odak-dashboard) | Web dashboard |
| [`odak-docs`](https://github.com/odak-analysis/odak-docs) | Submitted project documents |

## Contributing

Branch, commit, pull request and review rules: https://github.com/odak-analysis/.github/blob/main/CONTRIBUTING.md
