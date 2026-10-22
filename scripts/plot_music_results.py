"""Plot held-out musical task accuracy and audio ablations from saved results."""
import argparse
import json
from pathlib import Path

from samsone_clm.common import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=ROOT/'models/samsone_clm_music/evaluation_test.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/music/music_results.png')
    parser.add_argument('--title',default='Samsone–CLM music · fixed held-out test')
    args=parser.parse_args()
    report=json.loads(args.evaluation.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    structure={'tempo','meter','key_mode','key_tonic'}
    tasks=report['correct']['music_tasks']
    groups=[(sorted(set(tasks)-structure),'Perception',1.2),
            (sorted(set(tasks)&structure),'Musical structure',1)]
    groups=[group for group in groups if group[0]]
    fig,axes=plt.subplots(1,len(groups),figsize=(15,9) if len(groups)>1 else (10,9),
                          constrained_layout=True,squeeze=False,
                          gridspec_kw={'width_ratios':[group[2] for group in groups]})
    axes=axes[0]
    for ax,(names,title,_) in zip(axes,groups):
        y=np.arange(len(names))
        for offset,branch,label,color in [(-.25,'correct','Correct audio','#258d80'),
                                          (0,'random','Random audio','#ce9840'),
                                          (.25,'silence','Silence','#8767b6')]:
            values=[100*report[branch]['music_tasks'][name]['class_macro_top1'] for name in names]
            ax.barh(y+offset,values,height=.23,label=label,color=color)
        ax.set_yticks(y,[name.replace('_',' ')+f" (n={tasks[name]['samples']})" for name in names])
        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_xlabel('Semantic-class balanced Top-1 (%)')
        ax.set_title(title)
        ax.grid(axis='x',alpha=.2)
    axes[-1].legend(loc='upper right')
    fig.suptitle(args.title,fontsize=16)
    fig.supxlabel('Different tasks have different candidate counts. Small genre cohorts are exploratory.\n'
                  'Jamendo excerpt labels are weak supervision; MusicBench structure labels are automatic.',fontsize=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)
    print(args.output)


if __name__=='__main__':
    main()
