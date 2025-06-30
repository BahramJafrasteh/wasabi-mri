import pandas as pd
import numpy as np
from .utils import get_merged


import pandas as pd
import numpy as np

def load_uk_dataset(path_vol, path_qc=None, use_all_rois=False, adni_scale=None, nmax=None):
    """
    Loads and preprocesses brain volume data with optional QC and normalization.

    Parameters
    ----------
    path_vol : str
        Path to volume CSV file (must contain 'subject' and 'total intracranial').
    path_qc : str or None
        Path to QC CSV file. If provided, applies QC threshold on 'general grey matter'.
    use_all_rois : bool
        Whether to use all merged ROIs or hemisphere-summed subsets.
    adni_scale : float or None
        If provided, rescale global mean volume to match ADNI.
    nmax : int or None
        Max number of rows to load (for fair comparison, e.g. UKB subset).

    Returns
    -------
    vol_data : np.ndarray
        Preprocessed and scaled brain volumes [N, D].
    mean_scale : float
        Mean value used for scaling (for applying elsewhere).
    """

    data = pd.read_csv(path_vol)
    if nmax is not None:
        data = data.iloc[:nmax, :]

    if path_qc:
        qc = pd.read_csv(path_qc)
        if nmax is not None:
            qc = qc.iloc[:nmax, :]
        qc_filtered = qc[qc['general grey matter'] > 0.69]
        data = data.merge(qc_filtered[['subject']], on='subject', how='inner')

    vol = data['total intracranial']
    data.iloc[:, 1:] = data.iloc[:, 1:].div(vol, axis=0)


    if use_all_rois:
        data = get_merged(data)
        vol_data = data.iloc[:, 2:].values  # skip 'subject' and TIV
    else:
        vol_data = data.iloc[:, 34:68].to_numpy() + data.iloc[:, 68:102].to_numpy()

    mean_scale = np.mean(vol_data)
    if adni_scale is not None:
        vol_data = vol_data / mean_scale * adni_scale

    return vol_data, mean_scale



def load_adni_volumes(path="ADNI_NC_data.csv", use_all_rois=True):
    df = pd.read_csv(path)
    if use_all_rois:
        df = get_merged(df)
        df.drop(columns='subject', inplace=True)
        adni_vol = df.iloc[:, 3:].values
    else:
        adni_vol = df.iloc[:, 35:69].values + df.iloc[:, 69:103].values  # left + right
    adni_scale = np.mean(adni_vol)
    return adni_vol, adni_scale

def load_brainsyn_volumes(
    vol_path="brainsyn_vol.csv",
    qc_path="brainsyn_qc.csv",
    adni_scale=1.0,
    use_all_rois=True
):
    df = pd.read_csv(vol_path)
    qc = pd.read_csv(qc_path)
    qc_filtered = qc[qc['general grey matter'] > 0.69]
    df = df.merge(qc_filtered[['subject']], on='subject', how='inner')

    vol = df['total intracranial']
    df.iloc[:, 1:] = df.iloc[:, 1:].div(vol, axis=0)  # Normalize by TIV

    if use_all_rois:
        df = get_merged(df)
        brainsyn_vol = df.iloc[:, 2:].values
    else:
        # Use only a subset (e.g., cortical left + right ROIs)
        brainsyn_vol = df.iloc[:, 34:68].to_numpy() + df.iloc[:, 68:102].to_numpy()

    brainsyn_scale = np.mean(brainsyn_vol)
    brainsyn_vol = brainsyn_vol / brainsyn_scale * adni_scale  # global scale matching

    return brainsyn_vol
