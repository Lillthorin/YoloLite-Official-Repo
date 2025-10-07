import numbers
import torch

def build_scheduler(optimizer, cfg, steps_per_epoch):
    sch_cfg = cfg["training"].get("scheduler", None)
    if not sch_cfg:
        return None, "epoch"
    if isinstance(sch_cfg, str):
        sch_cfg = {"type": sch_cfg}
    elif isinstance(sch_cfg, bool):
        sch_cfg = {"type": "none"} if sch_cfg is True else None
        if sch_cfg is None:
            return None, "epoch"
    elif not isinstance(sch_cfg, dict):
        sch_cfg = {"type": "none"}

    sch_type = str(sch_cfg.get("type", "none")).lower()
    if sch_type in ("none", "off", "disable"):
        return None, "epoch"

    if sch_type == "cosine":
        T_max  = int(sch_cfg.get("t_max", cfg["training"]["epochs"]))
        eta_min = float(sch_cfg.get("min_lr", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        return scheduler, "epoch"

    if sch_type == "step":
        step_size = int(sch_cfg.get("step_size", 30))
        gamma     = float(sch_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, "epoch"

    if sch_type == "multistep":
        milestones = sch_cfg.get("milestones", [60, 80])
        gamma      = float(sch_cfg.get("gamma", 0.1))
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return scheduler, "epoch"

    if sch_type == "onecycle":
        max_lr = sch_cfg.get("max_lr", None)
        if max_lr is None:
            max_lr = [pg["lr"] for pg in optimizer.param_groups]
        elif isinstance(max_lr, numbers.Number):
            max_lr = [max_lr] * len(optimizer.param_groups)
        elif isinstance(max_lr, list) and len(max_lr) != len(optimizer.param_groups):
            max_lr = [pg["lr"] for pg in optimizer.param_groups]

        epochs = int(cfg["training"]["epochs"])
        pct_start  = float(sch_cfg.get("pct_start", 0.3))
        div_factor = float(sch_cfg.get("div_factor", 25.0))
        final_div  = float(sch_cfg.get("final_div_factor", 1e4))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=max(1, steps_per_epoch),
            pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div, anneal_strategy="cos"
        )
        return scheduler, "step"

    if sch_type == "plateau":
        factor   = float(sch_cfg.get("factor", 0.1))
        patience = int(sch_cfg.get("patience", 5))
        min_lr   = float(sch_cfg.get("min_lr", 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr, verbose=True
        )
        return scheduler, "plateau"

    print(f"[scheduler] Okänd typ '{sch_type}', scheduler avstängd.")
    return None, "epoch"