# GST-Hackathon-24
this project involves creating AI and machine learning (ML) solutions using a large dataset provided by the Goods and Services Tax Network (GSTN). 
The dataset contains about 900,000 records with 21 attributes. Using this data to build models that can predict certain outcomes and identify patterns that may help improve tax systems and detect fraud. 
The main goal of the system is to develop a model that can accurately predict target values for new, unseen data. 
The process begins with building a model that uses input data to make predictions. 
This model will be trained on a portion of the dataset to optimize its performance and minimize errors.
After training, it will be tested on new data to see how well it predicts outcomes.
A binary classification model has been developed that in turn, used various algorithms of machine learning to make predictions over the target class. 
Data were processed step by step, having started with null value imputation with LightGBM Imputer, followed by outlier handling with IQR and scaling of features along with oversampling techniques used for class balancing. 
The preprocessing steps of cleaning ensured that the data was clean and therefore ready for modeling. This may lead to greater accuracy and generalization. 
Machine-learning algorithms such as Logistic Regression, Decision Trees, and XGBoost were used. 
Each of them gave valuable insight into the strengths and weaknesses of different classification techniques. 
After all the training and evaluation, the best algorithm for our given classification task was XGBoost, giving better accuracy and precision.
Collectively, they provide a more balanced view of the performance of the models, and we can compare our effectiveness in terms of predicting the target variable. 
We may do fine-tuning, taking a closer look at a trade-off between precision and recall for reducing both false positives and false negatives. 
The exercise provided an overview of the overall process of binary classification from preparation of data to testing of a model. 
Results obtained indicate how machine learning models shape space and leave open further probing into predictive tasks in the field of data analytics.
