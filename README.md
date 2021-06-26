# Diabetic Retinopathy
Detection of Diabetic Retinopathy using Data Processing (Tensorflow 1.12.0 and Python 3.6.6)

- Editor used: Sublime Text 
- Shell used to run the code: Git Bash
- Libraries used: Pandas, Numpy & Tensorflow
- DR.csv is the main dataset, which is used for the training and testing stages of the model
- demo.csv is the prediction dataset, which is used as input for the prediction stage of the model after the training and testing stages are completed

Insight on DR.csv
- There are a total of 20 columns of data
- The first 19 columns serve as the features for the model
- The last (20th) column serves as the result, which the model predicts and trains itself on during the prediction and training+testing stages respectively. The predict.csv file does not contain the 20th/result column since the model is supposed to predict that result and generate the same as it's output.
- Column 1: Represents the binary result of quality assessment of the retinal scans, 0 = bad quality 1 = sufficient quality
- Coumn 2: Represents the binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack
- Column 3-8: Represents the results of MA detection. Each feature value stand for the number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively
- Column 9-16: Contains the same information as columns 3-8, but, for exudates. However, as exudates are represented by a set of points rather than the number of pixels constructing the lesions, these features are normalized by dividing the number of lesions with the diameter of the ROI to compensate different image sizes
- Column 17: Represents the euclidean distance of the center of the macula and the center of the optic disc to provide important information regarding the patient's condition. This feature is also normalized with the diameter of the ROI
- Column 18: Represents the diameter of the optic disc
- Column 19: Represents the binary result of the AM/FM-based classification
- Column 20: Represents the class label, 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of Diabetic Retinopathy
