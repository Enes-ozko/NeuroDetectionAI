import copy
import numpy as np
import torch
import torch.nn as nn

def train_one_fold(model, fold_data, cfg, device):
    train_loader = fold_data["train_loader"]
    val_loader   = fold_data["val_loader"]

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None
    no_improve = 0

    model.to(device)

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"  Epoch {epoch + 1:03d}/{cfg['epochs']} | loss={total_loss / len(train_loader):.4f} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"Early stopping(epoch {epoch + 1})")
                break

    model.load_state_dict(best_weights)
    return model, best_val_acc


def train_etape3(get_model_fn, folds, cfg, device):
    results = []
    for fold_data in folds:
        print(f"\nFold {fold_data['fold']}/{cfg['n_folds']}")
        model, best_val_acc = train_one_fold(get_model_fn(), fold_data, cfg, device)
        results.append({
            "fold": fold_data["fold"],
            "model": model,
            "best_val_acc": best_val_acc,
            "val_loader": fold_data["val_loader"],
        })

    accs = [r["best_val_acc"] for r in results]
    print(f"\nMoyenne val_acc : {np.mean(accs):.3f} +/-{np.std(accs):.3f}")
    return results