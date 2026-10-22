import sys
from samsone_clm.data import main

if __name__ == "__main__":
    if "--source-shards" in sys.argv or "--spike" in sys.argv:
        if "--spike" in sys.argv:
            sys.argv.remove("--spike")
        sys.argv.insert(1, "download")
        main()
    else:
        from samsone_clm.full_data import download_all
        download_all()
