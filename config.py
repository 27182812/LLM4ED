import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
parser.add_argument("--data_path", type=str, default="data/ED/")
parser.add_argument("--save_path", type=str, default="results/ED_chatgpt.txt")
parser.add_argument("--max_seq_length", type=int, default=320)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--few_shot', type=int, default=5)
parser.add_argument('--stage', type=int, default=1)
parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument('--apikey', type=str, default='')
    

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)

args = parser.parse_args()
print_opts(args)

model = args.model # "gpt-3.5-turbo", "text-davinci-003, "davinci"
data_path = args.data_path
save_path = args.save_path
max_seq_length = args.max_seq_length
temperature = args.temperature
few_shot = args.few_shot
stage = args.stage
device = torch.device("cuda" if args.cuda else "cpu")
apikey = args.apikey

