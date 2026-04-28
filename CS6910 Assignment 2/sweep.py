import wandb

sweep_config = {
    "method": "random",

    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },

    "program": "train.py",

    "parameters": {

        "learning_rate": {
            "values": [0.001, 0.0005, 0.0001]
        },

        "batch_size": {
            "values": [16, 32, 64]
        },

        "dropout": {
            "values": [0.2, 0.3]
        },

        "activation": {
            "values": ["relu", "gelu"]
        },

        "filters": {
            "values": [
                [32, 64, 128, 256, 512],
                [16, 32, 64, 128, 256]
            ]
        }
    }
}

sweep_id = wandb.sweep(
    sweep_config,
    project="cs6910_assignment2_partA"
)

print("Sweep ID:", sweep_id)