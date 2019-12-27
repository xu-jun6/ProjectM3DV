在进行检测时，将test.py，weight.h5，以及测试集数据test文件夹放在同一路径下。
直接运行test.py，它会读入权重文件weight.h5,并读入数据集进行预测。在同一路径下会输出一个预测文件submission.csv，这就是预测的数据。它取得的分数和我在leaderboard上的分数0.65747应该是相同的。

model_train.py文件含有使用的模型以及训练的过程。

medel2.py是尝试的第二个模型（这里并未使用）。
