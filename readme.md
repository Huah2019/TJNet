# File structure
    Dataset
        TrainDataset
        TestDataset
    TJNet
        models
            res2net50_v1b_26w_4s-3cf99910.pth -------Backbone network pre-trained model
        net
           Res2Net.py                         
           TJNet.py
        trainpth                              -------The training process files are saved here
        results                               -------The test files are saved here
        utils                                 -------Some tools
        results.txt                           -------Metrics are saved here
        Train.py                              -------Train model
        Test.py                               -------Test model
        Val.py                                -------Get Metrics
        config.ini						      -------Experimental parameter configuration
# pre-trained model
Res2Net50: https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth

TJNet: https://drive.google.com/file/d/1sxLEPsqOlppmCtJ58VegZ2F1jgYfrRU2/view?usp=sharing

# Dataset
TrainDataset:https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing

TestDataset:https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing
