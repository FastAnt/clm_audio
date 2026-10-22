"""Download official FMA archives with resume and published checksum verification."""
import argparse,hashlib,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from samsone_clm.common import ROOT,write_json
from samsone_clm.resources import require_resources

FILES={
'fma_large.zip':(100306112191,{'497109f4dd721066b5ce5e5f250ec604dc78939e'}),
'fma_metadata.zip':(358412441,{'f0df49ffe5f2a6008d7dc83c6915b31835dfe733'}),
'fma_medium.zip':(23825005356,{'c67b69ea232021025fca9231fc1c7c1a063ab50b','fe23d6f2a400821ed1271ded6bcd530b7a8ea551'}),
}
def download(name):
    require_resources()
    directory=ROOT/'raw/music/fma';directory.mkdir(parents=True,exist_ok=True)
    target=directory/name;partial=directory/(name+'.part')
    url='https://os.unil.cloud.switch.ch/fma/'+name
    if not target.exists():
        subprocess.run(['curl','--fail','--location','--retry','5','--retry-delay','5','--continue-at','-',
                        '--output',str(partial),url],check=True)
        partial.replace(target)
    size=target.stat().st_size
    if size!=FILES[name][0]:raise ValueError(f'Official archive length mismatch {name}: {size}')
    h1=hashlib.sha1();h256=hashlib.sha256()
    with target.open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):h1.update(block);h256.update(block)
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    if h1.hexdigest() not in FILES[name][1]:raise ValueError('Published FMA SHA1 mismatch: '+name)
    card=dict(official_url=url,bytes=size,sha1=h1.hexdigest(),sha256=h256.hexdigest(),verified=True)
    write_json(directory/(name+'.verified.json'),card)
    print(json.dumps(card),flush=True)
    return card

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--files',nargs='+',choices=list(FILES),default=['fma_metadata.zip','fma_medium.zip']);args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(download,args.files))
    write_json(ROOT/'raw/music/fma/downloads.json',results)
