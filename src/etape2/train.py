import copy
import numpy as np
import torch
import torch.nn as nn


def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):

    lam      = np.random.beta(alpha, alpha)
    idx      = torch.randperm(imgs.size(0), device=imgs.device)
    imgs_mix = lam * imgs + (1 - lam) * imgs[idx]
    return imgs_mix, labels, labels[idx], lam


def mixup_loss(loss_fn, logits, labels_a, labels_b, lam):
    return lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)


def train_one_fold(model, fold_data, cfg, device):
 
    train_loader = fold_data["train_loader"]
    val_loader   = fold_data["val_loader"]

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    loss_fn     = nn.BCEWithLogitsLoss()
    mixup_alpha = cfg.get("mixup_alpha", 0.2)

    model.to(device)

    best_val_acc = 0.0
    best_weights = None
    no_improve   = 0

    for epoch in range(cfg["epochs"]):

        # Entraînement
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            imgs_mix, la, lb, lam = mixup_batch(imgs, labels, mixup_alpha)

            optimizer.zero_grad()
            logits = model(imgs_mix)
            loss   = mixup_loss(loss_fn, logits, la, lb, lam)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds   = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == la.round()).sum().item()
            total   += la.size(0)

        train_acc = correct / total

        # Validation 
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                preds  = (torch.sigmoid(model(imgs)) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1:03d}/{cfg['epochs']} | "
            f"loss={total_loss/len(train_loader):.4f} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"Early stopping à l'epoch {epoch+1} | meilleure val_acc={best_val_acc:.3f}")
                break

    model.load_state_dict(best_weights)
    return model, best_val_acc


def train_etape2(get_model_fn, folds, cfg, device):

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
    print(f"\nMoyenne val_acc : {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"Meilleur fold : Fold {best['fold']} ({best['best_val_acc']:.3f})")

    return results