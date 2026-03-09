my_get_map.py是检测代码，运行之后会对可见光图像进行Yolo检测生成检测框，之后加载配准的高光谱进行ACE检测，并进行多模协同的检测。所有检测结果保存在map_out文件夹以及test文件夹。

multimode.py是多模检测运算的代码，被my_get_map调用。

predict.py可以生成保存有Yolo检测框的图像。
                             
map_out文件夹：其中detection-results是yolo检测结果，multimode-detection-results是多模协同结果，temp-results是保存中间结果（ace检测图等）。

test文件夹：origin是原始图像，predict是yolo检测结果图像，multimodal-predict是多模检测结果图像。

spectral_lib文件夹：构建的光谱库，光谱库的mat文件保存的是131*129的数组，其中第一行为背景均值mu_x_minus，最后一行数据为处理过的标准光谱向量sig，中间129*129的数据为inv_sigma的特征分解结果sqrtD*invU。

model_data文件夹：其中包含了训练好的yolo模型。

hyperspectral文件夹：高光谱数据。

VOCdevkit文件夹：可见光数据。



**********************************************************************************************************************************************************
                                                                                                                         结果对比示例
**********************************************************************************************************************************************************
Get map.
94.53% = aircraft AP 	||	score_threhold=0.1 : F1=0.67 ; Recall=100.00% ; Precision=50.00%
67.52% = car AP 	||	score_threhold=0.1 : F1=0.68 ; Recall=84.76% ; Precision=57.42%
95.96% = oiltank AP 	||	score_threhold=0.1 : F1=0.87 ; Recall=98.46% ; Precision=78.05%
94.66% = roof AP 	||	score_threhold=0.1 : F1=0.65 ; Recall=95.65% ; Precision=49.44%
100.00% = ship AP 	||	score_threhold=0.1 : F1=0.91 ; Recall=100.00% ; Precision=83.33%
mAP = 90.53%
Get map done.

Get multimodal map.
97.60% = aircraft AP 	||	score_threhold=0.1 : F1=0.85 ; Recall=100.00% ; Precision=74.07%
72.20% = car AP 	||	score_threhold=0.1 : F1=0.72 ; Recall=84.76% ; Precision=62.68%
95.96% = oiltank AP 	||	score_threhold=0.1 : F1=0.90 ; Recall=98.46% ; Precision=82.05%
95.93% = roof AP 	||	score_threhold=0.1 : F1=0.88 ; Recall=95.65% ; Precision=81.48%
100.00% = ship AP 	||	score_threhold=0.1 : F1=0.95 ; Recall=100.00% ; Precision=90.91%
mAP = 92.34%
Get map done.