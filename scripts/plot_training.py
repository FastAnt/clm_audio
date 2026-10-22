import argparse
from samsone_clm.monitor import render_png

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",type=int,choices=(1,2,3,4,5,6))
    parser.add_argument('--run',help='Explicit run directory, including remote and music runs')
    args=parser.parse_args()
    print(render_png(args.stage,args.run))
