from __future__ import annotations

import torch

from .config import NetworkConfig, QMCConfig, TrainConfig
from .train import evaluate_natural, evaluate_rule, train_natural, train_rule


def main() -> None:
    # Tiny run for infrastructure testing.  It is intentionally not a final
    # solve: the real run uses many more steps and validation nodes.
    train_cfg = TrainConfig(batch_size=64, steps=5, lr=1e-4, dtype=torch.float64, device="cpu")
    qmc_cfg = QMCConfig(n_train=16, n_val=64, seed=11)
    net_cfg = NetworkConfig(hidden_width=32, hidden_depth=2)

    natural_net, natural_log = train_natural(train_cfg=train_cfg, qmc_cfg=qmc_cfg, net_cfg=net_cfg, log_every=1)
    fixed_net, fixed_log = train_rule(
        natural_net,
        policy="fixed",
        train_cfg=train_cfg,
        qmc_cfg=qmc_cfg,
        net_cfg=net_cfg,
        log_every=1,
    )
    bottleneck_net, bottleneck_log = train_rule(
        natural_net,
        policy="bottleneck",
        train_cfg=train_cfg,
        qmc_cfg=qmc_cfg,
        net_cfg=net_cfg,
        log_every=1,
    )
    repair_aware_net, repair_aware_log = train_rule(
        natural_net,
        policy="repair_aware",
        train_cfg=train_cfg,
        qmc_cfg=qmc_cfg,
        net_cfg=net_cfg,
        log_every=1,
    )
    print("natural_log", natural_log)
    print("fixed_log", fixed_log)
    print("bottleneck_log", bottleneck_log)
    print("repair_aware_log", repair_aware_log)
    print("natural_eval", evaluate_natural(natural_net, train_cfg=train_cfg, qmc_cfg=QMCConfig(n_train=32, seed=12), n_states=32))
    print(
        "fixed_eval",
        evaluate_rule(
            fixed_net,
            natural_net,
            policy="fixed",
            train_cfg=train_cfg,
            qmc_cfg=QMCConfig(n_train=32, seed=13),
            n_states=32,
        ),
    )
    print(
        "bottleneck_eval",
        evaluate_rule(
            bottleneck_net,
            natural_net,
            policy="bottleneck",
            train_cfg=train_cfg,
            qmc_cfg=QMCConfig(n_train=32, seed=14),
            n_states=32,
        ),
    )
    print(
        "repair_aware_eval",
        evaluate_rule(
            repair_aware_net,
            natural_net,
            policy="repair_aware",
            train_cfg=train_cfg,
            qmc_cfg=QMCConfig(n_train=32, seed=15),
            n_states=32,
        ),
    )


if __name__ == "__main__":
    main()
