from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from tqdm import tqdm
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# pip install pysodmetrics

methods = []
methods.append('TJNet_best_camo')
methods.append('TJNet_best_chameleon')
methods.append('TJNet_best_cod10k')

data_names = []
data_names.append('CAMO')
data_names.append('CHAMELEON')
data_names.append('COD10K')
data_names.append('NC4K')

for method in methods:
    print("method: {}".format(method))
    for _data_name in data_names:
        mask_root = '../Dataset/TestDataset/{}/GT'.format(_data_name)
        pred_root = './results/{}/{}/'.format(method, _data_name)
        mask_name_list = sorted(os.listdir(mask_root))
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        print(results)
        file = open("results.txt", "a")
        file.write(method+' '+_data_name+' '+str(results)+'\n')
