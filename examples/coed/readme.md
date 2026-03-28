# CoED-GNN Node Classification

- Paper link: [https://arxiv.org/abs/2410.14109](https://arxiv.org/abs/2410.14109)
- Author's code repo: [https://github.com/hormoz-lab/coed-gnn](https://github.com/hormoz-lab/coed-gnn)

# Dataset Statics

| Dataset | # Nodes | # Edges | # Classes |
|---------|---------|---------|-----------|
| Cora    | 2,708   | 10,556  | 7         |

This reproduction uses the `Geom-GCN` 10 fixed splits on `Cora`.

## Files

- `examples/coed/coed_trainer.py`: training and evaluation entry for Cora node classification
- `examples/coed/geom_planetoid.py`: helper for loading `Geom-GCN` fixed splits
- `examples/coed/run_coed_cora.py`: lightweight launcher for the trainer
- `gammagl/models/coed.py`: CoED-GNN backbone model
- `gammagl/layers/conv/coed_conv.py`: CoED directional convolution layer

## Environment

```bash
/home/mr/venv/gammagl-py311-cpu
```

## Results

Run the reproduction with:

```bash
cd /home/mr/GammaGL-fork
source /home/mr/venv/gammagl-py311-cpu/bin/activate
python examples/coed/coed_trainer.py
```

Or:

```bash
cd /home/mr/GammaGL-fork
bash examples/coed/reproduce_cora.sh
```

The target reference result is:

```text
test acc: 86.41851 +/- 1.37720
```

The locally verified result is:

```text
test acc: 87.00201 +/- 1.43747
```

## Notes

- The implementation uses the GammaGL `Planetoid` Cora dataset and stores `Geom-GCN` split files under `data/planetoid/cora/geom-gcn/raw`.
- The default setup uses `hidden_dim=128`, `num_layers=2`, `lr=5e-4`, `weight_decay=1e-4`, `drop_rate=0.5`, `alpha=0.0`, `self_loop=True`, `normalize=False`, and `self_feature_transform=False`.
- A short smoke test with `--n_epoch 5` is only for pipeline verification and should not be used as the final reproduction result.
