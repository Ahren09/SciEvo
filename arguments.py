import os
import os.path as osp
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import const

def parse_args():

    parser = argparse.ArgumentParser(description="")
    # Parameters for Analysis

    parser.add_argument('--batch_size', type=int, default=256,
                        help="the batch size for models")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--data_dir', type=str,
                        default="data",
                        help="Location to store the processed dataset")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--do_visual', action='store_true',
                        help="Whether to do visualization")
    parser.add_argument('--embedding_dim', type=int, help="Step size for the scheduler", default=50)
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--embed_dim', type=int, default=100, help="Dimension of the generated embeddings.")
    parser.add_argument('--graph_backend', type=str, default="networkx", choices=["networkx", "rapids"], help="Dimension of the hidden layer.")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_name', type=str, choices=[const.WORD2VEC, const.GCN],  default=const.GCN)
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of processes")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Location to store the generated analytics "
                                                                          "or intermediate results")
    parser.add_argument('--save_every', type=int, default=20, help="Step size for the scheduler")
    parser.add_argument('--save_model', action='store_true', help="Whether to save the trained model")

    parser.add_argument('--start_year', type=int, default=None, help="Year to start downloading")
    parser.add_argument('--end_year', type=int,default=None, help="Year to end downloading")

    parser.add_argument('--load_from_cache', action='store_true', help="Whether to load the processed dataset from "
                                                                       "cache. ")
    parser.add_argument('--step_size', type=int, help="Step size for the scheduler", default=50)

    parser.add_argument('--feature_name', type=str, choices=["title", "abstract", "title_and_abstract"],
                        default='title')
    parser.add_argument('--tokenization_mode', type=str, choices=["unigram", "llm_extracted_keyword"], default='llm_extracted_keyword')
    parser.add_argument('--graphistry_personal_key_id', type=str, default='')
    parser.add_argument('--graphistry_personal_key_secret', type=str, default='')

    parser.add_argument('--min_occurrences', type=int, help="Minimum number of times a keyword needs to appear in the corpus",
                        default=5)

    args = parser.parse_args()

    args.data_dir = osp.expanduser(args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    return args
