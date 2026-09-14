import numpy as np
import nibabel as nib


def _affine_to_from_RAH(affine):

    # convert to/from nifti coordinate system
    rot_180 = np.identity(4, dtype=np.float32)
    rot_180[:2,:2] = [[-1,0],[0,-1]]

    return np.matmul(rot_180, affine)


def convert_to_nifti(vol):

    affine = _affine_to_from_RAH(vol.affine)
    nifti = nib.Nifti1Image(vol.values, affine)

    return nifti