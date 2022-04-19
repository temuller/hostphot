#!/usr/bin/env python

import wget
import tarfile
import os
import sys

try:
    import hostphot
except:
    pass

def download_dustmaps(mapsdir='.'):
    """ Downloads dust maps of Schlegel, Fikbeiner & Davis (1998).

    mapsdir : str, default '.'
        Directory where the directory with dust maps of Schlegel, Fikbeiner & Davis (1998)
        is going to be downloaded with the name 'sfddata-master/'. Default uses current
        directory.

    """

    sfdmaps_url = 'https://github.com/kbarbary/sfddata/archive/master.tar.gz'
    master_tar = wget.download(sfdmaps_url)

    # extract tar file under mapsdir directory
    tar = tarfile.open(master_tar)
    tar.extractall(mapsdir)
    tar.close()

    os.remove(master_tar)

if __name__ == '__main__':
    if len(sys.argv)==1:
        download_dustmaps()
    elif str(sys.argv[1])=='hostphot':
        mapsdir = hostphot.__path__[0]
        download_dustmaps(mapsdir)
    else:
        download_dustmaps(str(sys.argv[1]))
