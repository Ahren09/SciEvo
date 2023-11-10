import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="")
    # Parameters for Analysis

    parser.add_argument('--batch_size', type=int, default=256,
                        help="the batch size for models")
    parser.add_argument('--data_dir', type=str,
                        default="data",
                        help="Location to store the processed dataset")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--do_visual', action='store_true',
                        help="Whether to do visualization")
    parser.add_argument('--embedding_dim', type=int, help="Step size for the scheduler", default=50)
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_name', type=str, choices=["Word2Vec", "GConvGRU"],  default="GConvGRU")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of processes")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Location to store the generated analytics "
                                                                          "or intermediate results")
    parser.add_argument('--save_every', type=int, default=20, help="Step size for the scheduler")
    parser.add_argument('--save_model', action='store_true', help="Whether to save the trained model")
    parser.add_argument('--step_size', type=int, help="Step size for the scheduler", default=50)

    args = parser.parse_args()
    return args
