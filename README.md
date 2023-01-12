# Credit Card Fraud Detection

![amount by age](https://user-images.githubusercontent.com/95157802/211978280-1b2916cd-26e1-4ab5-8999-1710fdbed113.PNG)

The study aims to identify which characteristics of a credit card transaction are most important to determine whether it is fraudulent or legitimate. __Several new features will be engineered__ and derived from the existing data to capture important *customer and transaction details* as well as *customer purchase history and spending behaviour*.

The following key questions will be answered:

1. What features can be engineered to help enrich the data?
2. What is the amount distribution difference between fraudulent and legitimate transactions? 
3. Is there a particular age, profession and category that is more common in fraudulent transactions?
4. What features are most important in predicting whether a credit card transaction if fraudulent?

## This repo includes the following:

- "_Credit_Card_Fraud_Detection.ipynb_" - Jupyter notebook showcasing Python data pre-processing workflow of the raw CSV file, feature engineering, data manipulation, exploration and analysis, classification machine learning model training and evaluation.

Raw CSV file containing credit card transaction data sourced from [here](https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv).

## Insights

Althought the __Jupyter notebook is more thorough and captures the associated code and additional markdowns that help explain the project and thinking process__, below are some of the main insights and visualizations of this project:

### __Feature Engineering__

New features are engineered to capture __Credit Card Holder Details__, __Transaction Details__ and __Credit Card Holder Purchase Behaviour__.

> __Original DataFrame__
![original df](https://user-images.githubusercontent.com/95157802/211978392-358034e9-a319-48d5-b571-b68ab045072f.PNG)

> __Enriched DataFrame__
![head new df](https://user-images.githubusercontent.com/95157802/211979075-b2e0c1ce-595e-4705-8eb1-a3b106926fd3.jpg)
![describe new df](https://user-images.githubusercontent.com/95157802/211979103-0ebcea08-4d4d-4729-838a-12233412748e.jpg)


### __Credit Card Holder Details__

Firstly, we will engineer features related to credit card holder information:

- *card_id: Create unique identifier for rows which represent transactions made on the same card*

As there is currently no column in the DataFrame that represents transactions which belong to the same credit card, we will use other columns of personal details such as <code>dob</code>, <code>job</code> and <code>city</code> to derive a <code>card_id</code> unique identifier for the observations which belong to the same card. This will allow us to analyze purchase behaviour later on.<br> 

- *age: Determine card holder age*

By obtaining the difference between a person's date of birth <code>dob</code> and today's date, we can obtain a person's age. It is possible that people of certain age demographics are more susceptible to fraud.

### __Transaction Details__

Secondly, we will engineer features related to transaction details:

- *distance_from_merchant(km): Determine distance between card holder's location, and the merchant's location where a transaction has taken place*

Given the geographical location of a credit card holder, and location of the merchant, we can derive the distance between the two. There is a chance that transactions made at locations far from the card holder are suspicious.

- *day*, *month* and *year* (3 columns respectively): Extract the day, month and year of each transaction date*

The transaction date column will be used to extract the components of the date, which could later reveal For example, whether there is a time of the year where fraud is more common.

- *Check whether transaction is a whole number: *is_whole_number**

If a transaction has a whole number as opposed to numbers with decimals, this could also be an indicator of a suspicious transaction.

### __Credit Card Holder Purchase Behaviour__

Thirdly, we will engineer features related to a person' purchase behaviour:

- *avg_amount: Average transaction amount for all transactions of the same card*

- *avg_amt_30_day: Average monthly transaction amount of the same card over the past 30 days from the observation's date*

- *avg_amt_cat_30_day: Average transaction amount of the same card per category for the past 30 days from the observation's date*

The idea with each of these features is to uncover purchasing habits and behaviours. Any transactions which deviate and show discrepancies from the personal purchase behaviour captured in these new features could be an indicator that someone else is actually making the purchase, and could signify a fraudulent transaction.

### __Insights on "Age", "Job" and "Category" Features for Fraudulent Transactions__

![amount distributions](https://user-images.githubusercontent.com/95157802/211980145-ad520d53-9095-4f8f-80d9-c42364e74bee.PNG)
![amount by age](https://user-images.githubusercontent.com/95157802/211980176-7ceb85ba-9496-4bd1-bbe1-b47c777f000c.PNG)
![count fraud per job](https://user-images.githubusercontent.com/95157802/211982467-9d4462bd-2b54-42f6-bb94-219b93a2a57d.PNG)
![count of fraudulent transactions per category](https://user-images.githubusercontent.com/95157802/211980246-22652792-7e42-4fa9-a435-3a57f2d7a0fe.PNG)

- It appears __all age groups are at risk of becoming fraud victims__, and no real "one age group" is most susceptible or likely to be targetted. The interquartile range lies approximately between 38 and 65 which represents 50% of all observations, with a median of about 52.

- Among the top 20 job occupations that have the highest count of fraudulent transactions in the dataset, several of these professions are highly technical. However, this does not indicate that these jobs are most susceptible to fraud, as a single card could have been compromised and several fraudulent transaction made on that same card (which belongs to 1 person working 1 job).

- In terms of spending categories in fraudulent transactions, __grocery point of sale__, and __Internet shopping__ are the most frequently occurring categories, with around 400 fraudulent transaction counts each. These are then followed by __miscellaneous Internet shopping__, __point of sale shopping__ and __gas_transport__, with around 220, 180 and 150 counts respectively.

### __Insights on Machine Learning Model Developing and Testing__

![roc curve](https://user-images.githubusercontent.com/95157802/211980370-91172f6c-7a1a-4bce-affd-a27bb7037dd4.PNG)

![ml model performance df](https://user-images.githubusercontent.com/95157802/211986598-658e3277-6bd3-481b-90f2-34b99ea33b62.PNG)

- All models have over 99% accuracy but have varying precision, recall and AUC scores.
- Out of the 2 best models tested, the *GradientBoostingClassifier* has a Precision of 87.8% and Recall of 71.5%, while *RandomForestClassifier* has a Precision of 98.8% and Recall of 50.5%. While the *GradientBoostingClassifier* has successfully detected 71.5% of the positive (fraud) observations in the testing dataset, the *RandomForestClassifier* has only detected 50.5%.
- On the other hand, the *GradientBoostingClassifier* positive predictions were correct 87.8% of the time, while the *RandomForestClassifier* positive predictions were only correct 71.5% of the time.
- Despite having a lower AUC score and Precision, the *RandomForestClassifier* is better at detecting fraud as demonstrated by its Recall score, so it will be used moving forward. Due to the significant class imbalance, it is better to have a few incorrectly labelled non-fraudulent transactions so long as most or all fraudulent transactions were successfully detected by the algorithm, and then manually looking through the model's positively classified observations to rule out incorrectly labelled legitimate transactions.

### __Insights on the RandomForestClassifier Method (better learning method)__

![feature importance](https://user-images.githubusercontent.com/95157802/211986672-85fe5438-5954-46c3-b918-e928981f8c30.PNG)

![confusion matrix](https://user-images.githubusercontent.com/95157802/211982288-79f98f4d-e082-4bc4-ba09-5c5b24f3c99c.PNG)

- The final confusion matrix shows the labels predicted by the model, which inclue 368 True Positives, 101,339 True Negatives, 50 False Positives and 147 False Negatives.
- The most important features in the learning of the model are <code>amt</code>, <code>hour</code>, <code>category_grocery_pos</code>, <code>age</code> and <code>avg_amt_30_day</code>. Out of the model's 10 most important features, half of them were features engineered from the original dataset features.
- Most of the important features relate to either amounts of money spent, or the category of purchase, although the hour of the day and age of the cardholder are also very important.
- <code>amt</code> is the most important feature in the learning of this algorithm, which makes sense given the disparity in the distribution of values between fraudulent and legitimate transaction, as was identified earlier through the boxplot visualizations.

## __Conclusion__

This study has explored the topic of fraudulent credit card transactions through the lens of data science, utilizing a dataset containing both legitimate and fraudulent transactions to firstly engineer new features, secondly conduct exploratory data analysis and thirdly to develop and evaluate machine learning models to detect fraud.

__Feature engineering was employed to create several new features that describe meaningful aspects of the samples__, enriching the original data. These could then be used for visualization, and were important in the training of the machine learning models tested.

The final _GradientBoostingClassifier_ has a Recall score of 71.5%, which indicates it successfully detected 71.5% of the fraudulent observations in unseen data, the original purpose of this model.

The __feature which proved to be most important__ in the final model's classification prediction are the __amounts associated with the transaction__. Earlier EDA also reveals how there is a significant discrepancy between the distributions of the amount <code>amt</code> columns of the legitimate and fraudulent DataFrames, which is something the model likely considered to be significant.

The model does have room for improvement, however, especially to increase its ability to correctly predict more fraudulent transactions. Some strategies that could be experimented with involve: getting additional fraudulent transaction data that the model can learn from, fine tuning the model's hyper parameters through a grid search, or identifying and removing outliers from the amount column.
