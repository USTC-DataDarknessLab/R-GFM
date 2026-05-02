import argparse
import sys

import torch
from parser.parser_node_level import add_node_level_args
from trainers.node2graph_trainer import Node2GraphTrainer

def main():
    parser = argparse.ArgumentParser(description="Node2Graph Task")
    add_node_level_args(parser)
    args = parser.parse_args()
    print(args)
    acc_list = []
    trainer = Node2GraphTrainer(args)
    trainer.run()
    for i in range(10):
        test_acc = trainer.one_shot_finetune(seed=i)
        acc_list.append(test_acc)

    acc_tensor = torch.tensor(acc_list)
    mean_acc = acc_tensor.mean().item()
    std_acc = acc_tensor.std(unbiased=False).item()

    print(f"Test Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

if __name__ == "__main__":
    main()
