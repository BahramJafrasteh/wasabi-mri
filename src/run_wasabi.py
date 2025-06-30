from ADNI.results.src.metrics import compute_metrics
from ADNI.results.src.prepare_data import load_adni_volumes, load_brainsyn_volumes, load_uk_dataset
import numpy as np
def main():
    use_all_rois = True

    adni_vol, adni_scale = load_adni_volumes("ADNI_NC_data.csv", use_all_rois=use_all_rois)
    uk_vol, uk_scale = load_uk_dataset('ukbiobank_vol.csv', '/midtier/sablab/scratch/baj4003/pycharmprojects/asFCM/ADNI/ukbiobank_qc.csv',
                                       use_all_rois=True, adni_scale=adni_scale, nmax=1000)

    brainsyn_vol = load_brainsyn_volumes(
        vol_path="brainsyn_vol.csv",
        qc_path="brainsyn_qc.csv",
        adni_scale=adni_scale,
        use_all_rois=use_all_rois
    )
    score_uk_uk = compute_metrics(uk_vol, uk_vol, K=1000, criterion='wasabi')
    d1_uk_uk = np.array([np.mean(score_uk_uk[el]) for el in score_uk_uk.keys()])

    score_uk_brainsyn = compute_metrics(uk_vol, brainsyn_vol, K=1000, criterion='wasabi')


    d1_uk_b = np.array([np.mean(score_uk_brainsyn[el]) for el in score_uk_brainsyn.keys()])

    modesl = ['UK to UK', 'Brainsyn to UK']
    for i, d in enumerate([d1_uk_uk, d1_uk_b]):
        mult = 1e4
        print(f'Distribution: {modesl[i]}, Mean: {np.mean(d) * mult:.2f}, Std: {np.std(d) * mult:.3f}'
              )

if __name__ == "__main__":
    main()
