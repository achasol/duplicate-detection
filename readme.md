# Duplicate detection



1. TODO also keep track of F1-measure currently tracking only F1*-measure 
Precision & Recall 

2. TODO look at efficient fine-tuning of BERT model! 

3. Fix bug with already_found (cannot use product_id!) Since other pairs can exist (4 dups possible)! 

4. Test new BERT model performance using a perfect sim measure 

5. When performance is good start looking at Catboost/Classifier to fix that 


Likely that the Catboost model is overfitting (high performance in-sample but out of sample F1 scores quite poor..)
Maybe switch and try a random forest model (more sparse) 



potential problem unable to find any duplicates:
Pair quality and completeness 0 

1. Bad embeddings (retraining now)
2. Bad catboost model (check with gold labels)
3. Bad LSH parameters (vary some params)
4. Invalid draw XX (not the case, already validated) 

Look at the way the golden labels are used right now 
does not really make much sense 

Need to carefully look at definition of pair quality and pair completeness 