# Clustering Book Passages
This project focuses on the comparing different unsupervised machine learning techniques on its's ability to cluster passages based on its' authorship. 

### Data 
In this dataset, 200 passages containing 150 words each were randomly taken from 5 different books in the NLTK Corpus library (a total of 1000 instances). The five different books include:
 - The Man Who Was Thursday - G. K. Chesterton (1908)/ Chesterton-Thursday 
 - Melville – Moby Dick
 - Stories – E. Bryant (1900-2000) / Bryant-stories  
 - Paradise Lost – John Milton (1667) / Militon-Paradise 
 - Hamlet – William Shakespeare (1609) /Shakespeare-Hamlet
 
### Preprocessing 
* Words are lemmatized 
* Stops are removed in the final model to compare results via Spacy 
* PCA: Dimensionality reduction technique applied with BOW and TF-IDF to reduce 12000 features down to 300 features, to accelerate the training( this step is not necessary with Word2Vec due to its low dimensionality)  
### Feature Engineering 
* BOW  (with PCA) 
* TF-IDF (with/without PCA)
* Word2Vec  
### Models implemented 
* K-Means 
* Hierachial Clustering 
* Gaussian Mixtures 
* LDA 
### Model results based on number of cluster 
#### K-Means and EM Method 
The number of cluster and it's impact on the quality was compared. The following are sample results for TFIDF+PCA, but this was also repeated for BOW+PCA and Word2Vec. 

The silhouette score  was calculated for different number of clusters for both K-means and EM method. For silhouette score, the maximum value is usually the best result. For K-means, the optimum cluster number is 6, where as for EM, the optimum number of cluster is 5. In both cases, this optimum cluster number is close to the empirical number of data sources, which is 5 books. When the same analysis was repeated for BOW and Word2Vec, the optimum silhouette score was with number of clusters were actually close to 2 or 8, which was way off (see notebook). This demonstrates that clustering using TFIDF features can yield patterns that are similar to the data. 

For the elbow method of K-means, the point where slope begin to change is also 5, indicating optimum cluster.  
For EM method, the BIC and AIC values for the number of clusters was plotted. Usually lower BIC and AIC score means better cluster results. Here, they have coinciding results because as the cluster number increase, the AIC score decreases. However, as the BIC score increases. However, around 5 to 6 cluster is when BIC score is somewhat stabilized and when AIC score is low. Based on this analysis, the optimum number of clusters for Kmeans with TFIDF and PCA was also found to be around 5 to 6. 

![image](https://user-images.githubusercontent.com/29676594/115329622-8b3e9980-a160-11eb-87df-275c4b907c3f.png)
#### LDA and Hierarchical Method 
A similar analysis on the coherence score was repeated for TFIDF with LDA. Here the k means number of topics chosen, and greater coherence score is better. One odd thing here is that using four or six topics yields a higher coherence score than using five topics, despite that the empirically the data is aggregated from five topics(authors).  This leads to the conclusion that LDA may not able to correctly model the topics of this data. LDA was not further applied for the next steps of the analysis. 

For hierarchical clustering, the dendrogram of these clusters were examined. From the five clusters, each cluster looks to be equal size of approximately 200 data points, which is similar to the size of the empirical clusters.  

![image](https://user-images.githubusercontent.com/29676594/115330050-57b03f00-a161-11eb-8800-c6e71f98724d.png)

### Annotated Passage Results and Kappa Score 
#### Passage Annotation 
It's important to acknowledge that there is no annotated labels in true unsupervised machine learning. However, since this project is for exploratory purposes, these labels where the book are from) are avilable. In this case, the hidden labels are used to annotate each cluster's origin. This annotation process is done by matching all the passages in one cluster to its empirical label (book source). Then, the most frequently occurring label becomes the annotation of that cluster. As an example, the following figure represents this process for TF-IDF/PCA-KMeans.



![image](https://user-images.githubusercontent.com/29676594/115331024-2769a000-a163-11eb-8d0f-899aec3b8ec0.png)

Note that similar approach like this is also commonly used in semi-supervised machine learning, where small amount of annotated labels are used to identify data points that are close to the cluster center, and the labels are applied to the rest of the dataset.

#### Annotated clusters reveals that TF-IDF is the superior feature engineerng choice for clustering different book passages

The annotated cluster provide insights on the clustering algoritm. The figure below demonstrates the K-mean clusters that are presented with three different feature engineering steps, and was repeated for EM and Hierarchial. 

With TF-IDF/PCA, most of the clusters are only from one single book source, each cluster appear to have very low overlay. With BOw, while most of the cluster also came from one book source, the cluster’s impurity seems to decrease compared to TFIDF. However, when word 2 vec is used, the clustering quality greatly reduces. In fact, majority of the clustering group does not come from a single book anymore.In this case, word 2 vec performed terribly. 

The zero, first and econd component of these features are also plotted, each cluster are color coded based on the annotation. . It’s very obvious here that the cluster centers for TFIDF is very far apart, the centers for BOW is closer but still somewhat apart, and all the cluster centers for Word 2 Vec is very overlayed. This provide insight that TFIDF with PCA yield very discriminatory features that clustered well, whereas word 2 vec yield features that are too similar and can not be easily clustered. 


![image](https://user-images.githubusercontent.com/29676594/115331105-5253f400-a163-11eb-8bda-c8c44bcced1b.png)

This was process was also applied to HC and EM, and in all cases, TF-IDF performed the best, BOW performed somewhat behind, and Word2Vec greatly suffered. 


#### Kappa score demonstrates that choice of feature engineering affects results far greater than the model choice 
This is the aggregated model evaluation , where the top is the kappa score and the bottom is the number of errors from comparing the empirical labels with the annotated labels. The optimal result is TFIDF+PCA with K-means, with second one being TFIDF+PCA with EM method.Kappa score in general is highest for TFIDF transformed vector, followed by BOW features, with the word2vec features scoring very poorly. K-means and EM models also scored similarly, where as the kappa score for all  hierarchical clustering falls slightly behind. However, the feature engineering step affects the clustering quality far greatly than the model choice. 

##### Kappa Score

![image](https://user-images.githubusercontent.com/29676594/115331836-aad7c100-a164-11eb-9961-287936b65156.png)

##### Number of Errors  

![image](https://user-images.githubusercontent.com/29676594/115332023-0904a400-a165-11eb-9551-09aefda4c0fa.png)


#### Affect of removing stop words
Stops words were removed from passages using SPACY to determine if this improved the final model of TF-IDF with K-Means. 
 
For PCA+TF-IDF features with K-means, the errors associated with passages that did not correspond to the annotated clusters are analyzed. The biggest error is with Chesterton Thursday and Bryant stories both getting misclassified as Moby Dick. The most common words in these passages are words such as “the”, “and” or “pron” , all are lemmatized word for pronouns. 

Upon this analysis, the stop word are removed to see if this would reduce such error. However, the kappa score has dropped after stop word are removed. Although removing stop words reduced the highest errors like clustering Chesterton Thursday passages into Moby Dick, this step introduced a lot of new errors such as misclassifying clustering Moby Dick into Bryant stories.  It’s possible that different authors may have different writing styles, and certain stop words may be used more often than others. Taking the stop words out can therefore also introduces these errors.    

![image](https://user-images.githubusercontent.com/29676594/115332423-c8f1f100-a165-11eb-8f25-f53908169798.png)



### Conclusions 
* This project tuned and compared 4 different algorithms -  3 feature selections to overcome the challenge of clustering passages from five different books 
* The optimal combination was K-Means with TF-IDF/PCA. Kappa score of 0.9587
* Optimum feature engineering method is TF-IDF with PCA, followed by BOW-PCA, then W2V
* EM and K-Means resulted in similar kappa scores, whereas hierarchical clustering resulted in slightly lower kappa scores 
* Although not removing stop words can include too many non-discriminatory features making it difficult to cluster certain passage, taking out stop words also reduces some important discriminatory features  



### Authors:
#### Team Lead : Xun Xun Shi
####  Additional Contributors : Lasya Bhaskara, Tulika Shukla 
