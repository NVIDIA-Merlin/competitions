# SIGIR 2021 E-Commerce Workshop Data Challenge - Purchase Intent Prediction Task

This online appendix is created as a supplement for our system description paper for 2021 SIGIR eCom Data Challenge submitted to [SIGIR eCom'21](https://sigir-ecom.github.io/index.html). 

## 1. Task Description

[Sigir Coveo Data Challenge](https://sigir-ecom.github.io/data-task.html) presented two tasks for the participants: 

1. A recommendation task, where a model is shown k events at the start of a session, and it is asked to predict future product interactions in the same session;
2. An intent prediction task, where a model is shown a session containing an add-to-cart event, and it is asked to predict whether the item will be bought before the end of the session.

Here, we focus on the second one - Purchase Intent Prediction Task. Our latest submission to LB before Stage 2 submissions were closed resulted in `3.6340530847665` weighted Micro-F1 score, and our best single model submission gave a score of <b>`3.6363084325068`</b> (see Section 8 for details). In the following sections we will describe our data preparation, feature engineering and selection, model building and hypertuning processes. Finally, we conclude our online appendix with a discussion section which gives some insights about model ensembling and prediction analysis.

## 2. Data Preparation and Exploration 

Our goal at this step was to create sequential features for user sessions and generate classification label for model training and evaluation. Due to the nature of the Task 2, our final goal is to create a binary classification model which is expected to predict whether the shopper will buy a product X or not in a given session containing an add-to-cart event for this product. 

For this data challenge, a session-level dataset, containing 10M product interactions over 57K different products (Section 3) is generated from the four tables `browsing`, `search`, `sku_content` and `intention_test` provided during the competition. Since the provided train set does not include the label column, we first created `is_purchased` label column which consists of binary values: 0 and 1. 

In the following subsections, we explain how interactions within the same session are aggregated, how train and validation sets were created for 5-fold Cross-Validation and finally we present different sampling strategies of train and validation sets to maximize the early prediction goal of Task 2. 


### 2.1. Data Preprocessing

During our preprocessing and feature engineering pipeline, the critical stage was to create the dataframe with features at session level. Before building the sessions, we noticed duplicated interactions which occur at the same time in the same session. For those duplicated events, we removed the rows where the event_type is `pageview`. Then we followed the same data augmentation strategy we defined for Task 1 to obtain a session as a sequence of three types of events : `product-action`, `click` and  `pageview`. Moreover we defined two strategies for grouping the interactions of the same session. The first version (Preprocessing 1) kept all interactions in the session without removing repetitions, while in the second version (Preprocessing 2) we removed duplicated interactions following the same strategy defined in Task 1. In both versions, we generated the same sequential features as the ones defined in Task 1.
 
As labels for training data were not provided, we created the label `is-purchased` following three steps:
1. Truncate interactions before the first purchase event
2. Filter out sessions without an `add-to-cart` event.
3. Assign label `1` to sessions with a purchase event, and label `0` to the others.
 
We also defined the `nb_after_add` variable, required by the scoring metric, for training sessions by taking the closest value in [0, 2, 4, 6, 8, 10] to the actual number of actions happening after the first add-to-cart event.


### 2.2. Cross-validation Split

After we created the `session_interaction` dataset, we reserved the last three weeks of the training set for validation. Then we prepared data for 5-fold cross validation by randomly assigning a session in Train, Validation or Test set to a fold in the range of [1, 5].


### 2.3. Strategies for creating Train / Validation Sets to Mimic the Prediction Objective

The distribution of sessions with respect to the number of actions recorded after the first add-to-cart event (AC) varies between Train, Validation and Test sets. In fact, the test set was manually engineered to emphasize the early prediction of the user's purchase intention by providing more sessions with fewer actions after the AC. Our goal was to test five different strategies to find the trade-off between getting a Train set with enough relevant information for fitting a model able to distinguish between positive and negative labels and at the same time building a Validation set close to Test set distribution to evaluate the early prediction power of the trained model. Figure 1. shows the five distributions defined below. 

* `Original distribution`: we kept all sessions with their actual `nb_after_add` value 
* `Resample Train and Validation sets`: we truncated longer sessions of Train and Validation sets to have same distribution as Test data. 
* `Original Train set and resampled Validation set`: we kept Train sessions with their original length and truncated Validation sessions to have same distribution as Test ones. 
* `Balanced Validation set`: we kept Train sessions with their original length and truncated Validation sessions to have a balanced distribution between Train and Test data. 
* `Duplicated sessions with different cuts after (AC)`:  We duplicated each session in Train and Validation by truncating it at different points in [0, 2, 4, 6, 8, 10] events after AC.

<p align="center">
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/resampling_strategies.png" width="800" height="600"/>
  <br>
  <font size="1">Figure 1. Distrbution of sessions with respect to number of interactions after the (AC) event.</font>
</p>


## 3. EDA Analysis of Preprocessed Table of Sessions

EDA is always an important part of any ML/DL pipeline. For initial EDA about the training set we refer the reader to Task 1 section of the paper. In this section, we analyze the  profile of `session_interaction` dataset that includes a subset of pre-processed sessions with add-to-cart events. We plotted couple of figures here to give us better understanding of sessions with a purchase and those with a cart-abandonment.

Figure 2. shows that sessions with purchase or cart-abandonment have the same length distributions meaning that total number of unique interactions is not enough to predict the user's intention to purchase the first add-to-cart (AC) product. To get a more fine-grained relation between the purchase intention and the number of unique interactions, we plot in Figure 3. the distribution of interactions length before and after the AC event for both classes. From this figure, we could notice that there are more interactions after the AC than before for sessions with a purchase event. While there is an equal distribution of the number of interations before and after the AC for cart-abandonment sessions. As the objective of the competition is to detect early predictions using the 5 subgroups of `nb_after_add` events considered after the first add-to-cart action, we plot in Figure 4. the distribution of the binary classes within each subgroup. Following the conclusion of Figure 3. we observe that sessions with larger `nb_after_add` contains more purchase sessions.   

<p align="center">
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/train_session_length_dist.png" width="600" height="300" />
  <br>
  <font size="1">Figure 2. Train set session length distribution based on purchased label</font>
</p>

<p align="center">
 <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/dist_actions_before_after_AC.png" width="600" height="400">
  <br>
  <font size="1">Figure 3. Distribution of number of interactions before and after first AC event.</font>
</p>

<p align="center">
 <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/nb_after_add_train.png" width="600" height="300">
  <br>
  <font size="1">Figure 4. Nb after add for train with respect to purchase label.</font>
</p>


The table below shows the general statistics for certain features based on the preprocessed session interactions train and validation datasets. 

 Table 1. Preprocessed session interactions dataset statistics (mean, median, stddev).

<table class="table-table">
<thead><tr class="table-firstrow"><th>&nbsp;</th><th colspan=2>Train set</th><th colspan=2>Validation set</th><th colspan=2>Validation set resampled</th></tr></thead><tbody>
 <tr><th>Features</th><td>Is_purchased = 0</td><td>Is_purchased = 1</td><td>Is_purchased = 0</td><td>Is_purchased = 1</td><td>Is_purchased = 0</td><td>Is_purchased = 1</td></tr>
 <tr><td>nb_after_add</td><td>(4.72, 4.00, 3.78)</td><td>(5.92, 6.00, 3.31)</td><td>(4.49, 4.00, 3.72)</td><td>(5.75, 6.00, 3.30)</td><td>(1.89, 0.00, 2.83)</td><td>(2.42, 2.00, 3.05)</td></tr>
 <tr><td>position of the add event</td><td>(5.18, 3.00, 5.78)</td><td>(5.21, 4.00, 5.22)</td><td>(4.83, 3.00, 5.44)</td><td>(5.04, 4.00, 5.00)</td><td>(4.87, 3.00, 5.36)</td><td>(5.04, 4.00, 5.00)</td></tr>
 <tr><td>nb_of_add_events</td><td>(1.34, 1.00, 0.98)</td><td>(1.32, 1.00, 0.88)</td><td>(1.34, 1.00, 0.97)</td><td>(1.34, 1.00, 0.91)</td><td>(1.22, 1.00, 0.58)</td><td>(1.21, 1.00, 0.57)</td></tr>
 <tr><td>nb of different categories</td><td>(1.47, 1.00, 0.99)</td><td>(1.36, 1.00, 0.84)</td><td>(1.41, 1.00, 0.90)</td><td>(1.33, 1.00, 0.82)</td><td>(1.30, 1.00, 0.66)</td><td>(1.22, 1.00, 0.55)</td></tr>
 <tr><td>nb of different main categories</td><td>(1.17, 1.00, 0.44)</td><td>(1.13, 1.00, 0.38)</td><td>(1.15, 1.00, 0.42)</td><td>(1.12, 1.00, 0.37)</td><td>(1.12, 1.00, 0.35)</td><td>(1.08, 1.00, 0.29)</td></tr>
 <tr><td>nb of clicks</td><td>(0.22, 0.00, 0.84)</td><td>(0.29, 0.00, 0.87)</td><td>(0.25, 0.00, 0.89)</td><td>(0.34, 0.00, 1.01)</td><td>(0.23, 0.00, 0.77)</td><td>(0.30, 0.00, 0.81)</td></tr>
 <tr><td>nb_queries</td><td>(0.47, 0.00, 1.19)</td><td>(0.64, 0.00, 1.27)</td><td>(0.46, 0.00, 1.16)</td><td>(0.64, 0.00, 1.36)</td><td>(0.46, 0.00, 1.16)</td><td>(0.64, 0.00, 1.36)</td></tr>
 <tr><td>nb of samples</td><td>136462</td><td>38111</td><td>32084</td><td>7928</td><td>32084</td><td>7928</td></tr>
</tbody></table>


## 4. Feature Engineering (FE) for Tabular Models

### 4.1. Feature creation 

To prepare features for tabular models, we aggregated the sequence of interactions taking into account three different cut-points: considering the whole sequence, the sequence of interactions happened before the first AC event and the interactions occurred after the AC. From these three sequences, we created five categories of features (see Table 2): 
  - Session-level variables,  
  - the statistics of interactions before and after the AC, 
  - the description of the product added to cart,  
  - the intra-similarity of the session,
  - the surrounding interactions of the AC event.

The Intra-similarity of the session is defined using the cosine metric for products’ embeddings and the Jaccard similarity index for the main categories of products. We defined three similarity scores: 
  - The cosine similarity between the AC product and all other interacted items.
  - The Jaccard index between the category of AC product and all other categories. 
  - The Jaccard index between all categories.


<p> Table 2. Features created for model training.</p>
 
<table class="tableizer-table">
<thead><tr class="tableizer-firstrow"><th>Category</th><th>List of variables</th></tr></thead><tbody>
 <tr><td>Session</td><td>session_length', 'nb_unique_interactions', 'nb_queries'</td></tr>
 <tr><td>Add-to-cart</td><td>add_product_id', 'add_nb_interactions', 'add_has_been_detailed','add_has_been_removed', 'add_has_been_viewed','add_has_been_searched', 'add_has_been_clicked', 'add_category_hash', 'add_main_category', 'add_price', 'add_relative_price', 'add_relative_price_main'</td></tr>
 <tr><td>Interaction statistics</td><td>nb_add_before', 'nb_add_after', 'nb_detail_before','nb_detail_after', 'nb_remove_before', 'nb_remove_after', 'nb_view_before', 'nb_view_after' , 'nb_click_before', 'nb_click_after'</td></tr>
 <tr><td>First and last Interactions features</td><td>"product_url_id_list_after', 'event_type_list_after', 'product_action_list_after', 'category_list_after', 'price_list_after', 'relative_price_list_after', 'product_url_id_list_before', 'event_type_list_before', 'product_action_list_before', 'category_list_before', 'price_list_before', 'add_category_hash', 'add_main_category', 'add_price',   'add_relative_price',  'add_relative_price_main' </td></tr>
 <tr><td>Similarity features</td><td>'mean_sim_desc', 'std_sim_desc', 'mean_sim_img', 'std_sim_img', 'mean_sim_desc_before', 'std_sim_desc_before', 'mean_sim_img_before', 'std_sim_img_before', 'mean_sim_desc_after', 'std_sim_desc_after', 'mean_sim_img_after', 'std_sim_img_after', 'main_category_similarity_general', 'main_category_similarity_add'</td></tr>
</tbody></table>


### 4.2. Target Encoding 

Target Encoding (TE) is an encoding technique that replaces a categorical value with the mean of the target variable. With TE, we create numerical features in places of categorical features and feed them to the model. It is a simple and effective categorical feature encoding technique, particulary if we deal with high cardinality categorical features. More information about TE can be found [here](https://medium.com/rapids-ai/target-encoding-with-rapids-cuml-do-more-with-your-categorical-data-8c762c79e784).

We applied the TE technique to different categorical columns in a trial and error fashion, but in our 2nd place solution, only `product_id` column, which can be seen as a high cardinality column, was target encoded. The other categorical features were label encoded first and then they were fed to the XGB model. To apply TE, we used [Merlin NVTabular](https://github.com/NVIDIA/NVTabular) library which enables us to perform accelerated feature engineering and preprocessing on GPU(s). 

It is worth noting here that we applied target encoding with out-of-the-fold (OOF) strategy as a part of our training script. In doing so, we created folds both for training and validation sets, then we applied the `fit()` method using 4 folds from training set and collected statistics, and then we transformed both the out-of-the fold validation and in-fold train set.

## 5. Prediction Models

For our classification models we developed XGBoost [1] and DLRM [2] models. XGBoost (XGB), a scalable tree boosting algorithm, has been widely adapted by data science practitioners for classification tasks and achieved state-of-the-art results on many machine learning challenges over the last couple of years. DLRM [2] is a promising deep-learning model particulary developed for Recommender Systems problems and it makes use of both categorical and numerical inputs.

### 5.1. Hyperparameter Tuning

We conducted a bayesian hyperparameter optimization using the package Optuna [3]. Particularly, we performed 100-trial hyperparameter tuning for XGB model using the following hyperparameters and search space.

<p> Table 3. Hyperparameter space for XGB model.</p>
<table class="table-table">
<thead><tr class="tableizer-firstrow"><th>Hyperparameter name</th><th>Search Space</th><th>Sampling Distribution</th></tr></thead><tbody>
 <tr><td>num_round</td><td>[10, 500]</td><td>int_uniform (step 10)</td></tr>
 <tr><td>max_depth</td><td>[2, 20]</td><td>int_uniform (step 2)</td></tr>
 <tr><td>learning_rate</td><td>[0.01, 0.8]</td><td>log_uniform</td></tr>
 <tr><td>scale_pos_weight</td><td>[1, 4]</td><td>discrete_uniform(step 0.5)</td></tr>
 <tr><td>reg_lambda</td><td>[1, 10]</td><td>int_uniform (step 1)</td></tr>
 <tr><td>subsample</td><td>[0.2, 1]</td><td>discrete_uniform(step 0.2)</td></tr>
 <tr><td>colsample_bytree</td><td>[0.2, 1]</td><td>discrete_uniform(step 0.2)</td></tr>
 <tr><td>eval_metric</td><td>AUC</td><td>-</td></tr>
 <tr><td>objective</td><td>binary:logistic</td><td>-</td></tr>
 <tr><td>tree_method</td><td>gpu_hist</td><td>-</td></tr>
 <tr><td>predictor</td><td>gpu_predictor</td><td>-</td></tr>
</tbody></table>

After hyperparameter tuning, we trained the XGB model with the following hyperparameters' values.

<p> Table 4. Final Hyperparameter Values for XGB model.</p>
<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameter</th><th>Value</th></tr></thead><tbody>
 <tr><td>num_folds</td><td>5</td></tr>
 <tr><td>num_round</td><td>20</td></tr>
 <tr><td>max_depth</td><td>6</td></tr>
 <tr><td>learning_rate</td><td>0.5295173477</td></tr>
 <tr><td>reg_lambda</td><td>6</td></tr>
 <tr><td>colsample_bytree</td><td>0.4</td></tr>
 <tr><td>subsample</td><td>1</td></tr>
 <tr><td>scale_pos_weight</td><td>1</td></tr>
</tbody></table>

In our experiments we observed that DLRM is prone to overfitting after the 1st epoch, therefore we added regularization penalties in the model [4], which helped reducing overfitting to a certain degree. The following table shows the hyperparameter values used in training of the DLRM models. In the interest of time, we could not perform extensive hyperparameter tuning but before deciding the values below, we trained DLRM models with different learning rate values (e.g., 0.01, 0.005, 0.001), L1 and L2 penalties (e.g., 1e-6, 1e-5, 1e-4, 1e-3), epochs (e.g., [1, 5]), and embedding dimension size (e.g., 32 and 64). 

<p> Table 5. Hyperparameters for DLRM model.</p>
<table class="table-table">
<thead><tr class="table-firstrow"><th>Hyperparameters</th><th>Value</th></tr></thead><tbody>
 <tr><td>epochs</td><td>3</td></tr>
 <tr><td>L1_penalty</td><td>1e-3</td></tr>
 <tr><td>L2_penalty</td><td>1e-3</td></tr>
 <tr><td>learning_rate</td><td>0.001</td></tr>
 <tr><td>emb_dropout_rate</td><td>0.01</td></tr>
 <tr><td>dropout_rate</td><td>0.01</td></tr>
 <tr><td>emb_dim</td><td>64</td></tr>
 <tr><td>batch_size</td><td>4096</td></tr>
</tbody></table>

## 6. Feature selection : Random swapping 

The feature engineering process generated 98 features. You can see these features in the `cart_features_98.txt` file under the featurs folder. In order to reduce the large set of variables, we used a random noise swapping technique to select the features most relevant to the classification task. This technique consists of training an XGB model with all the features. Then, during inference, we iteratively select one feature and randomly shuffle its values to compute the CV accuracy. If the accuracy drops, we retain the swapped feature in the selected features set otherwise we drop it as it does not affect the accuracy. 

## 7. Ensembling the Models

The models we trained and used in the ensemble are XGBoost [1] and DLRM [2]. We applied five primary strategies when ensembling the models. The ensemble results are available in Table 4 at Section 8.3.

* Majority voting: The predictions with the majority consent from the models. 
* At least 2 votes: A test data point is labeled with a positive (1) class if at least two models labeled that data point as positive.  
* Keep all positive predictions: If any model classifies a test data point as positive then it is labeled as positive.
* Mean prediction value: The mean of the prediction values from all the models were calculated and with a threshold value of 0.5 a label was assigned. 
* Weighted mean prediction value: In this strategy, weighted prediction values were used for each of the models based on their LB scores. The mean of the weighted prediction values from all the models was calculated and the labels were assigned with a threshold value of 0.5. 


## 8. Results and Discussion 

After challenge ended, the submission system was kept open, so we got chance to test our ideas that would have resulted in higher LB score than our 2nd place score. We performed different strategies to create train and validation sets, and ensemble different models trained and evaluated with those sets.
 
### 8.1. Table with Different Single Models Scores 

In the Table 6 below, we report the LB and local CV scores with different models and sampling strategies. From line 2 to 6, we tested the different sampling strategies using XGB model. Keeping the original train set without truncating longer sessions leads to the best LB score. This confirms the conclusion of the EDA analysis section suggesting that interactions after the first AC are relevant to purchase intention prediction. In the `Section 8.3.2` we conduct a prediction analysis to better understand the performance of the best single model.

<p> Table 6. Single models' results</p>
<table class="table-table">
<thead><tr class="table-firstrow"><th colspan=1>Model</th><th colspan=1>Preprocessing</th><th colspan=1>Distribution strategy</th><th colspan=1>Feature set</th><th colspan=1>CV score</th><th colspan=1>LB score</th></tr></thead><tbody>

<tr><td>XGB</td><td>1</td><td>Original Train set and resampled Validation set</td><td>Selected features</td><td>3.837139793884</td><td>3.6340530847665</td></tr>

<tr><td>XGB</td><td>2</td><td>Original distribution</td><td>All 98 features</td><td>3.95232754338948</td><td><b>3.63473879683627</b></td></tr>

 <tr><td>XGB</td><td>2</td><td>Resampled Train and Validation sets</td><td>All 98 features</td><td>3.84617066289227</td><td>3.63220840883373</td></tr>

<tr><td>XGB</td><td>2</td><td>Original Train set and resampled Validation set</td><td>All 98 features</td><td>3.82705418722994</td><td>3.6255265035838</td></tr>

<tr><td>XGB</td><td>2</td><td>Balanced Validation set</td><td>All 98 features</td><td>3.8771796638453</td><td>3.62606032855852</td></tr>

<tr><td>XGB</td><td>2</td><td>Duplicated sessions with different cuts after (AC)</td><td>All 98 features</td><td>3.53892968880628</td><td>3.62821528284864</td></tr>

 <tr><td>DLRM<sup>*</sup></td><td>2</td><td>Original distribution</td><td>Selected features</td><td>3.9369424912343</td><td><b>3.63464047403249</b></td></tr>

<tr><td>DLRM<sup>**</sup></td><td>2</td><td>Original distribution</td><td>All 98 features</td><td>3.93082390829393</td><td>3.63338211440125</td></tr>
</tbody></table>

<sup>*</sup> DLRM model that resulted in the best score in the table above trained with the selected features given in `cat_features_DLRM.txt` and `num_features_DLRM.txt`files. 
<br>
<sup>**</sup> This DLRM model was trained with embedding dimension of 32. The 98 features can be found in the features directory.


From Table 6, we can observe that XGB results in higher score with all 98 features, whereas DLRM performs better with the selected features. Both models generated the highest score with non-truncated train dataset. 

### 8.2. Our 2nd Place LB Score 

Before Stage 2 submissions were closed our best score of `3.6340530847665` was achieved using a single hypertuned XGB model trained with sessions with repeated interactions (preprocessing 1) and a subset of features including the 5 last interactions in the session features (id, event and action types and hierarchy category), the type of interactions with add-to-cart product and the number of unique interactions and queries within the session. 

### 8.3. Best single model score 

The results below are from the best single XGB model that scored 3.63473879683627 on LB after competition was ended.

#### 8.3.1. Selected Features and their Importance 

The Figure 5 shows the subset of relevant features selected using random swapping and their relative importance given by the XGB model. We notice that interactions happening after the AC event are the most important to predict cart-abandonment. Another set of features relevant to the prediction is the interactions the user had with the AC product. 

<p align="center">
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/importance_of_selected_features.png" width="600" height="300" />
  <br>
  <font size="1">Figure 5. Importance of the features selected via feature swapping.</font>
</p>
 

### 8.3.2. Prediction Analysis

In the upper-left plot of Figure 6, we visualize the micro f1_score for each group of `nb_after_add`. The highest accuracy score is achieved for sessions with 0 or one event after the add to cart (AC) while the accuracy stabilizes for groups of 2, 4, 6 and 8 interactions after the AC event and drops sharply for sessions with more than 10 interactions after the AC. To better understand these scores, we plot in the upper-right figure the f1-score of positive and negative classes for each subgroup. For `nb_after_add==0` the model has a very high classification score for the negative class but fails to detect the purchase events. This discrepancy between both classes is not showcased in the micro f1_score as the positive class is only representing 6% of the subgroup sessions (see Figure 4.). For `nb_after_add==10`, we got a higher score of positive classes but as they occur more often in the subgroup 10, the micro f1_score is lower than for `nb_after_add==0`.  From these two plots, we can conclude that the model is able to distinguish between purchase and cart-abandonment for sessions with 2 or 9 actions after the AC event. However, it is much harder to predict the purchase intention either for sessions without any information after the add-to-cart or for the ones with longer interactions after the AC.  Given the difference in class distribution and model’s scores between the 6 subgroups, we trained 6 XGB models, with the parameters of the best XGB reported in Table 3, one for each subgroup. The LB score improves significantly from `3.63473879683627` to <b>`3.6363084325068`</b> suggesting that each group should be trained separately to learn its specificities.

<p align="left">
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/weighted_f1_nb_after_add.png" width="400" height="300" />
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/pos_neg_f1_score.png" width="400" height="300" />
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/wighted_f1_unique_interactions.png" width="400" height="300" />
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/images/wighted_f1_unique_interactions.png" width="400" height="300" />
    <br>
  <font size="1">Figure 6. Prediction analysis of the best single XGB model. </font>
</p>


#### 8.3.1.4. Ensemble results
 
Table 7 below shows the ensemble results obtained with different ensemble strategies (see Section 7) and models.  We have observed the models were predicting a limited number of the positive class. We were interested to see how the ensemble performed if we increase the positive class prediction. To test this idea we employed various strategies to get a different number of positive predictions. Although we have performed various ensembling strategies, Table 4 presents some notable ensemble and their scores. Due to time constraints and limited number of submissions in stage 2,  we could not test most of our ensemble strategies on the public LB during the competition. However, after the LB freeze, we achieved our best score (3.636871273) for the Purchase Intent Prediction Task.

<table>
 <p>Table 7: Ensemble scores</p>
  <tr>
    <td>Used Models</td>
    <td>Ensemble Strategy</td>
    <td>LB Score</td>
    <td># Positive Predictions</td>
  </tr>
  <tr>
   <td>XGB Balanced +  XGB WT + XGB NT</td>
   <td>Majority voting</td><td>3.633197414</td><td>131</td>
  </tr>
  <tr>
   <td>XGB Balanced +  XGB WT + XGB NT</td>
   <td>Keep all positive predictions</td><td>3.635348244</td><td>590</td>
  </tr>
  <tr>
   <td>XGB Balanced +  XGB WT + XGB NT + DLRM</td>
   <td>Majority voting</td><td>3.633946626</td><td>95</td>
  </tr>
  <tr>
   <td>XGB Balanced +  XGB WT + XGB NT + DLRM</td>
   <td>At least 2 positive votes</td><td><b>3.636871273</b></td><td>145</td>
  </tr>
  <tr>
   <td>XGB Balanced +  XGB WT + XGB NT + DLRM</td>
   <td>Keep all positive predictions</td><td>3.633891263</td><td>651</td>
  </tr>
  <tr>
   <td>XGB Balanced +  XGB WT + XGB NT + DLRM</td>
   <td>Mean prediction value</td><td>3.635604559</td><td>651</td>
  </tr>
</table>

 - XGB Balanced denotes the model trained with a balanced validation set (see section 2.3). 
 - XGB WT denotes the model trained with truncated dataset.
 - XGB NT denotes the model trained with no data truncation.

# REFERENCES

- [1] Tianqi Chen and Carlos Guestrin, XGBoost: A Scalable Tree Boosting System, June 2016, online-available: https://arxiv.org/pdf/1603.02754.pdf
- [2] Maxim Naumov, et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems", May 2019, online-available: https://arxiv.org/pdf/1906.00091.pdf 
- [3] Optuna - A hyperparameter optimization framework.  online-available: https://optuna.org/
- [4] tf.keras.regularizers.Regularizer, https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer
