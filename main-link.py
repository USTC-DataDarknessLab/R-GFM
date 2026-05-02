import argparse

from parser.parser_edge_level import add_edge_level_args
from trainers.edge2graph_trainer import Edge2GraphTrainer

def main():
    parser = argparse.ArgumentParser(description="Edge2Graph Task")
    add_edge_level_args(parser)
    args = parser.parse_args()
    print(args)
    trainer = Edge2GraphTrainer(args)
    trainer.run()

if __name__ == "__main__":
    main()
