import argparse
from samsone_clm.music_data import prepare_jamendo, prepare_nsynth, prepare_musicbench, write_dataset

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['jamendo','nsynth','musicbench'],required=True)
    parser.add_argument('--notes-per-instrument',type=int,default=20)
    args=parser.parse_args()
    rows=(prepare_jamendo() if args.dataset=='jamendo' else
          prepare_nsynth(args.notes_per_instrument) if args.dataset=='nsynth' else prepare_musicbench())
    print(write_dataset(rows,args.dataset+'.jsonl'),flush=True)
