"""Upload a local history.json to wandb retroactively."""
import json
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, help="Path to history.json")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--project", default="Stackelberg-Pythia160M")
    parser.add_argument("--group", default=None)
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    run = wandb.init(project=args.project, name=args.run_name, group=args.group,
                     resume="never")

    train = history["train"]
    val   = history["val"]
    val_by_step = {s: {"val/loss": l, "val/ppl": p}
                   for s, l, p in zip(val["step"], val["loss"], val["ppl"])}

    for i, step in enumerate(train["step"]):
        log = {
            "train/ce_loss":   train["ce"][i],
            "train/ce_ema":    train["ce_ema"][i],
            "train/div_loss":  train["div"][i],
            "train/leader_ce": train["leader_ce"][i],
        }
        if step in val_by_step:
            log.update(val_by_step[step])
        wandb.log(log, step=step)

    for step, metrics in val_by_step.items():
        if step not in train["step"]:
            wandb.log(metrics, step=step)

    wandb.finish()
    print(f"Uploaded {len(train['step'])} train steps and {len(val['step'])} val steps.")

if __name__ == "__main__":
    main()
