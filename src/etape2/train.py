import copy
import numpy as np
import torch
import torch.nn as nn

from src.etape2.model import freeze_bn


# Mixup 

def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):

    if alpha <= 0:
        return imgs, labels, labels, 1.0

    lam    = np.random.beta(alpha, alpha)
    bs     = imgs.size(0)
    idx    = torch.randperm(bs, device=imgs.device)
    imgs_a = imgs
    imgs_b = imgs[idx]
    labels_a = labels
    labels_b = labels[idx]
    imgs_mix = lam * imgs_a + (1 - lam) * imgs_b
    return imgs_mix, labels_a, labels_b, lam


def mixup_loss(loss_fn, logits, labels_a, labels_b, lam):
    return lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)


def smooth_labels(labels: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    return labels * (1 - epsilon) + (1 - labels) * epsilon


def tta_predict(model, imgs_pil_list, tta_transform, n: int, device: str) -> np.ndarray:

    all_probs = []
    for _ in range(n):
        batch = torch.stack([tta_transform(img) for img in imgs_pil_list]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


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

    loss_fn      = nn.BCEWithLogitsLoss()
    mixup_alpha  = cfg.get("mixup_alpha", 0.2)
    label_smooth = cfg.get("label_smoothing", 0.05)

    model.to(device)

    best_val_acc  = 0.0
    best_weights  = None
    no_improve    = 0

    for epoch in range(cfg["epochs"]):

        model.train()
        freeze_bn(model)

        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            labels_smooth = smooth_labels(labels, label_smooth)

            imgs_mix, la, lb, lam = mixup_batch(imgs, labels_smooth, mixup_alpha)
            la = la.to(device)
            lb = lb.to(device)

            optimizer.zero_grad()
            logits = model(imgs_mix)
            loss   = mixup_loss(loss_fn, logits, la, lb, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds   = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == la.round()).sum().item()
            total   += la.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        all_probs = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                probs  = torch.sigmoid(model(imgs))
                preds  = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                all_probs.extend(probs.cpu().squeeze().tolist())

        val_acc   = val_correct / val_total
        all_probs = np.array(all_probs)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:03d}/{cfg['epochs']} | "
            f"loss={total_loss/len(train_loader):.4f} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | "
            f"prob_mean={all_probs.mean():.3f} | lr={current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stopping epoch {epoch+1} | best val_acc={best_val_acc:.3f}")
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
    print(f"  Moyenne val_acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"  Meilleur fold   : Fold {best['fold']} ({best['best_val_acc']:.3f})")

    return results