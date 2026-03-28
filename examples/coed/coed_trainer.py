"""CoED-GNN node classification trainer for Cora on GammaGL."""

import argparse
import importlib.util
import os
import random
import sys

os.environ.setdefault("TL_BACKEND", "torch")

import numpy as np
import tensorlayerx as tlx

from gammagl.mpops import unsorted_segment_sum
from gammagl.utils import mask_to_index
from geom_planetoid import load_planetoid_with_geom_splits


def _load_local_coed_model():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "gammagl", "models", "coed.py")
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location("coed_model_local", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CoEDModel


CoEDModel = _load_local_coed_model()


class AdamLike:
    """A lightweight Adam optimizer used to avoid backend version conflicts."""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = {}
        self.v = {}

    def zero_grad(self, params):
        for param in params:
            if getattr(param, "grad", None) is not None:
                param.grad.zero_()

    def step(self, params):
        self.step_count += 1
        beta1_correction = 1.0 - self.beta1 ** self.step_count
        beta2_correction = 1.0 - self.beta2 ** self.step_count

        for idx, param in enumerate(params):
            grad = getattr(param, "grad", None)
            if grad is None:
                continue

            if idx not in self.m:
                self.m[idx] = tlx.zeros_like(param)
                self.v[idx] = tlx.zeros_like(param)

            grad_to_use = grad
            if self.weight_decay != 0.0:
                grad_to_use = grad_to_use + self.weight_decay * param

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * grad_to_use
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (grad_to_use * grad_to_use)

            m_hat = self.m[idx] / beta1_correction
            v_hat = self.v[idx] / beta2_correction
            update = self.lr * m_hat / (tlx.sqrt(v_hat) + self.eps)
            param.data.copy_(param.data - update)


def set_seed(seed):
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)


def collect_trainable_weights(module):
    """Collect trainable parameters recursively from a TLX module tree."""
    weights = []

    for weight in getattr(module, "_parameters", {}).values():
        if weight is not None and getattr(weight, "requires_grad", False):
            weights.append(weight)

    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            weights.extend(collect_trainable_weights(child))

    return weights


def clone_trainable_state(module, prefix=""):
    """Clone the current trainable state for early stopping restoration."""
    state = {}

    for name, weight in getattr(module, "_parameters", {}).items():
        if weight is not None and getattr(weight, "requires_grad", False):
            state[prefix + name] = weight.detach().clone()

    for child_name, child in getattr(module, "_modules", {}).items():
        if child is not None:
            state.update(clone_trainable_state(child, prefix=prefix + child_name + "."))

    return state


def restore_trainable_state(module, state, prefix=""):
    """Restore a previously cloned trainable state."""
    for name, weight in getattr(module, "_parameters", {}).items():
        key = prefix + name
        if weight is not None and key in state:
            weight.data.copy_(state[key])

    for child_name, child in getattr(module, "_modules", {}).items():
        if child is not None:
            restore_trainable_state(child, state, prefix=prefix + child_name + ".")


def row_normalize_features(x, eps=1e-12):
    """Apply row-wise feature normalization."""
    row_sum = tlx.reduce_sum(x, axis=1, keepdims=True)
    row_sum = tlx.maximum(row_sum, tlx.ones_like(row_sum) * eps)
    return x / row_sum


def get_edge_index_and_theta(edge_index):
    """Build the fuzzy edge list and its initial phase angles."""
    src = tlx.convert_to_numpy(edge_index[0]).tolist()
    dst = tlx.convert_to_numpy(edge_index[1]).tolist()

    edges = [(int(u), int(v)) for u, v in zip(src, dst) if u != v]
    edge_set = set(edges)

    triu_symm_edges = []
    triu_dir_edges = []
    tril_dir_edges = []

    for u, v in edges:
        if u < v:
            if (v, u) in edge_set:
                triu_symm_edges.append((u, v))
            else:
                triu_dir_edges.append((u, v))
        elif u > v and (v, u) not in edge_set:
            tril_dir_edges.append((u, v))

    triu_symm_edges = sorted(set(triu_symm_edges))
    triu_dir_edges = sorted(set(triu_dir_edges))
    tril_dir_edges = sorted(set(tril_dir_edges))

    if triu_symm_edges:
        if not triu_dir_edges and not tril_dir_edges:
            processed_edges = triu_symm_edges
            theta = [np.pi / 4.0] * len(triu_symm_edges)
        else:
            processed_edges = triu_dir_edges + tril_dir_edges + triu_symm_edges
            theta = [0.0] * (len(triu_dir_edges) + len(tril_dir_edges)) + [np.pi / 4.0] * len(triu_symm_edges)
    else:
        processed_edges = triu_dir_edges + tril_dir_edges
        theta = [0.0] * len(processed_edges)

    edge_index_fuzzy = tlx.convert_to_tensor(np.array(processed_edges, dtype=np.int64).T, dtype=tlx.int64)
    theta = tlx.convert_to_tensor(np.array(theta, dtype=np.float32), dtype=tlx.float32)
    return edge_index_fuzzy, theta


def get_fuzzy_laplacian(edge_index, theta, num_nodes, edge_weight=None, add_self_loop=False):
    """Construct normalized directional edge weights for CoED message passing."""
    senders = edge_index[0]
    receivers = edge_index[1]

    if edge_weight is None:
        edge_weight = tlx.ones((tlx.get_tensor_shape(senders)[0],), dtype=tlx.float32)

    theta = tlx.cast(theta, tlx.float32)
    edge_weight = tlx.cast(edge_weight, tlx.float32)
    cos_sq = tlx.cos(theta) ** 2
    sin_sq = tlx.sin(theta) ** 2

    conv_senders = tlx.concat([senders, receivers], axis=0)
    conv_receivers = tlx.concat([receivers, senders], axis=0)
    out_weight = tlx.concat([cos_sq * edge_weight, sin_sq * edge_weight], axis=0)
    in_weight = tlx.concat([sin_sq * edge_weight, cos_sq * edge_weight], axis=0)

    if add_self_loop:
        self_loops = tlx.arange(start=0, limit=num_nodes, dtype=tlx.int64)
        ones = tlx.ones((num_nodes,), dtype=tlx.float32)
        conv_senders = tlx.concat([conv_senders, self_loops], axis=0)
        conv_receivers = tlx.concat([conv_receivers, self_loops], axis=0)
        out_weight = tlx.concat([out_weight, ones], axis=0)
        in_weight = tlx.concat([in_weight, ones], axis=0)

    deg_senders = tlx.reshape(unsorted_segment_sum(out_weight, conv_senders, num_segments=num_nodes), (-1,)) + 1e-12
    deg_receivers = tlx.reshape(unsorted_segment_sum(in_weight, conv_senders, num_segments=num_nodes), (-1,)) + 1e-12

    deg_inv_sqrt_senders = tlx.where(deg_senders < 1e-11, tlx.zeros_like(deg_senders), tlx.pow(deg_senders, -0.5))
    deg_inv_sqrt_receivers = tlx.where(
        deg_receivers < 1e-11,
        tlx.zeros_like(deg_receivers),
        tlx.pow(deg_receivers, -0.5),
    )

    ew_src_to_dst = (
        tlx.gather(deg_inv_sqrt_senders, conv_senders)
        * out_weight
        * tlx.gather(deg_inv_sqrt_receivers, conv_receivers)
    )
    ew_dst_to_src = (
        tlx.gather(deg_inv_sqrt_receivers, conv_senders)
        * in_weight
        * tlx.gather(deg_inv_sqrt_senders, conv_receivers)
    )

    conv_edge_index = tlx.stack([conv_senders, conv_receivers], axis=0)
    conv_edge_weight = (tlx.reshape(ew_src_to_dst, (-1, 1)), tlx.reshape(ew_dst_to_src, (-1, 1)))
    return conv_edge_index, conv_edge_weight


def calculate_acc(logits, y, idx):
    """Calculate node classification accuracy on indexed nodes."""
    pred = tlx.gather(tlx.argmax(logits, axis=-1), idx)
    label = tlx.gather(y, idx)
    return float(tlx.reduce_mean(tlx.cast(pred == label, tlx.float32)))


def resolve_dataset_path(dataset_path):
    """Resolve a local Planetoid cache path before attempting any download."""
    candidates = [
        os.path.abspath(dataset_path),
        "/home/mr/GammaGL-fork/data/planetoid",
        "/home/mr/GammaGL/data/planetoid",
    ]
    for candidate in candidates:
        raw_dir = os.path.join(candidate, "cora", "raw")
        if os.path.exists(raw_dir):
            return candidate
    return os.path.abspath(dataset_path)


def main(args):
    """Train and evaluate CoED-GNN on the 10 Geom-GCN splits of Cora."""
    tlx.set_device("CPU")
    dataset_path = resolve_dataset_path(args.dataset_path)

    dataset, graph = load_planetoid_with_geom_splits(
        root=dataset_path,
        name=args.dataset,
        num_splits=args.geom_splits,
    )

    if args.normalize_features:
        graph.x = row_normalize_features(graph.x)

    edge_index, theta = get_edge_index_and_theta(graph.edge_index)
    edge_weight = tlx.ones((tlx.get_tensor_shape(edge_index)[1],), dtype=tlx.float32)
    conv_edge_index, conv_edge_weight = get_fuzzy_laplacian(
        edge_index=edge_index,
        theta=theta,
        num_nodes=graph.num_nodes,
        edge_weight=edge_weight,
        add_self_loop=args.self_loop,
    )

    split_test_accs = []
    for split_id in range(args.geom_splits):
        train_idx = mask_to_index(graph.train_mask[:, split_id])
        val_idx = mask_to_index(graph.val_mask[:, split_id])
        test_idx = mask_to_index(graph.test_mask[:, split_id])

        data = {
            "x": graph.x,
            "edge_index": conv_edge_index,
            "edge_weight": conv_edge_weight,
            "num_nodes": graph.num_nodes,
            "train_idx": train_idx,
        }

        for run in range(args.runs):
            set_seed(args.seed + split_id * 97 + run)

            model = CoEDModel(
                feature_dim=dataset.num_node_features,
                hidden_dim=args.hidden_dim,
                num_class=dataset.num_classes,
                num_layers=args.num_layers,
                alpha=args.alpha,
                drop_rate=args.drop_rate,
                normalize=args.normalize,
                self_feature_transform=args.self_feature_transform,
                name="CoED",
            )

            optimizer = AdamLike(lr=args.lr, weight_decay=args.weight_decay)
            train_weights = collect_trainable_weights(model)

            best_val_acc = 0.0
            best_test_acc = 0.0
            bad_counter = 0
            best_state = None

            for epoch in range(1, args.n_epoch + 1):
                model.set_train()
                optimizer.zero_grad(train_weights)
                logits = model.forward(data["x"], data["edge_index"], data["edge_weight"], data["num_nodes"])
                train_logits = tlx.gather(logits, data["train_idx"])
                train_y = tlx.gather(graph.y, data["train_idx"])
                loss = tlx.losses.softmax_cross_entropy_with_logits(train_logits, train_y)
                loss.backward()
                optimizer.step(train_weights)

                model.set_eval()
                logits = model.forward(data["x"], data["edge_index"], data["edge_weight"], data["num_nodes"])
                val_acc = calculate_acc(logits, graph.y, val_idx)
                test_acc = calculate_acc(logits, graph.y, test_idx)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_state = clone_trainable_state(model)
                    bad_counter = 0
                else:
                    bad_counter += 1

                if epoch % args.print_freq == 0 or epoch == 1:
                    print(
                        "split {:02d} run {:02d} epoch {:04d} loss {:.4f} val {:.4f} best_test {:.4f} patience {}/{}".format(
                            split_id,
                            run,
                            epoch,
                            float(loss.item() if hasattr(loss, "item") else loss),
                            val_acc,
                            best_test_acc,
                            bad_counter,
                            args.patience,
                        )
                    )

                if bad_counter >= args.patience:
                    break

            if best_state is not None:
                restore_trainable_state(model, best_state)
            model.set_eval()
            logits = model.forward(data["x"], data["edge_index"], data["edge_weight"], data["num_nodes"])
            best_test_acc = calculate_acc(logits, graph.y, test_idx)
            split_test_accs.append(best_test_acc)
            print("split {:02d} run {:02d} best test acc: {:.5f}".format(split_id, run, best_test_acc * 100.0))

    mean_test = float(np.mean(split_test_accs) * 100.0)
    std_test = float(np.std(split_test_accs) * 100.0)
    print("test acc: {:.5f} +/- {:.5f}".format(mean_test, std_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CoED-GNN classification reproduction on Cora with GammaGL/TensorLayerX."
    )
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--dataset_path", type=str, default="./data/planetoid")
    parser.add_argument("--geom_splits", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--self_loop", dest="self_loop", action="store_true")
    parser.add_argument("--no_self_loop", dest="self_loop", action="store_false")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--normalize_features", dest="normalize_features", action="store_true")
    parser.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    parser.add_argument("--self_feature_transform", dest="self_feature_transform", action="store_true")
    parser.add_argument("--no_self_feature_transform", dest="self_feature_transform", action="store_false")
    parser.set_defaults(
        self_loop=True,
        normalize=False,
        normalize_features=False,
        self_feature_transform=False,
    )
    main(parser.parse_args())
