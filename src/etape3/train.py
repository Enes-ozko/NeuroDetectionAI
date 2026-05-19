import copy
import numpy as np
import torch
import torch.nn as nn

from src.etape3.model import freeze_bn


def train_one_fold(model, fold_data, cfg, device):
    train_loader = fold_data["train_loader"]
    val_loader   = fold_data["val_loader"]

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("T_max", cfg["epochs"]),
        eta_min=cfg["lr"] * 0.01,
    )

    loss_fn        = nn.CrossEntropyLoss()
    best_val_acc   = 0.0
    best_weights   = None
    no_improve     = 0
    unfreeze_done  = False
    unfreeze_epoch = cfg.get("unfreeze_epoch", 8)

    model.to(device)

    for epoch in range(cfg["epochs"]):

        if epoch == unfreeze_epoch and not unfreeze_done:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["lr"] * 0.1,
                weight_decay=cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.get("T_max", cfg["epochs"]) - unfreeze_epoch,
                eta_min=cfg["lr"] * 0.001,
            )
            unfreeze_done = True
            print(f"  Backbone dégelé (epoch {epoch + 1})")

        model.train()
        freeze_bn(model)

        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                preds  = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc    = val_correct / val_total
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch + 1:03d}/{cfg['epochs']} | "
            f"loss={total_loss / len(train_loader):.4f} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | "
            f"lr={current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"Early stopping epoch {epoch + 1} | best val_acc={best_val_acc:.3f}")
                break

    model.load_state_dict(best_weights)
    return model, best_val_acc


def train_etape3(get_model_fn, folds, cfg, device):
    results = []

    for fold_data in folds:
        print(f"\nFold {fold_data['fold']}/{cfg['n_folds']}")

        model, best_val_acc = train_one_fold(get_model_fn(), fold_data, cfg, device)

        results.append({
            "fold":         fold_data["fold"],
            "model":        model,
            "best_val_acc": best_val_acc,
            "val_loader":   fold_data["val_loader"],
        })

    accs = [r["best_val_acc"] for r in results]
    print(f"\nMoyenne val_acc : {np.mean(accs):.3f} +/-{np.std(accs):.3f}")

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"Meilleur fold: Fold {best['fold']} ({best['best_val_acc']:.3f})")

    return results