import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="")
    # Parameters for Analysis

    parser.add_argument('--batch_size', type=int, default=256,
                        help="the batch size for models")

    parser.add_argument('--data_dir', type=str,
                        default="data",
                        help="Location to store the processed dataset")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Location to store the generated analytics "
                                                                          "or intermediate results")
    parser.add_argument('--save_model', action='store_true', help="Whether to save the trained model")
    parser.add_argument('--do_visual', action='store_true',
                        help="Whether to do visualization")
    parser.add_argument('--n_procs', type=int, default=16,
                        help="Number of processes")

    args = parser.parse_args()
    return args
