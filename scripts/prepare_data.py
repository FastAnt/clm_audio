import sys
from samsone_clm.data import main

if __name__ == "__main__":
    if "--samples" in sys.argv or "--spike" in sys.argv:
        if "--spike" in sys.argv:
            sys.argv.remove("--spike")
        sys.argv.insert(1, "prepare")
        main()
    else:
        from samsone_clm.full_data import prepare_all
        prepare_all(sys.argv[1:])
