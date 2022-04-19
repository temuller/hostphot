import unittest
import os
import hostphot
from hostphot.cutouts import download_multiband_images
from hostphot.local_photometry import multi_local_photometry

class TestHostPhot(unittest.TestCase):

    def download_dustmaps(self):
        mapsdir = hostphot.__path__[0]

        # check if files already exist locally
        dust_files = [os.path.join(mapsdir,
                    'sfddata-master',
                    f'SFD_dust_4096_{sky}gp.fits') for sky in ['n', 's']]
        mask_files = [os.path.join(mapsdir,
                    'sfddata-master',
                    f'SFD_mask_4096_{sky}gp.fits') for sky in ['n', 's']]
        maps_files = dust_files + mask_files
        existing_files = [os.path.isfile(file) for file in mask_files]

        if not all(existing_files)==True:
            # download dust maps
            sfdmaps_url = 'https://github.com/kbarbary/sfddata/archive/master.tar.gz'
            master_tar = wget.download(sfdmaps_url)
            # extract tar file under mapsdir directory
            tar = tarfile.open(master_tar)
            tar.extractall(mapsdir)
            tar.close()
            os.remove(master_tar)

    def test_local_phot(self):
        sn_name = 'SN2006hx'
        ra = 18.488792
        dec = 0.371667
        z = 0.045500
        survey = 'PS1'

        self.download_dustmaps()
        download_multiband_images(sn_name, ra, dec, survey=survey)
        multi_local_photometry([sn_name], [ra], [dec], [z],
                                ap_radius=4, survey=survey)

if __name__ == '__main__':
    unittest.main()
