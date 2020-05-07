# Practical Guide: Using Product Reviews to Support Internal Customer Operations


## Overview

In the following blog post, we're going to explore how to use AWS to scale data science experiments, with a particular focus on building Text Classifiers for supporting different functions within an organisation.

This blog post is supported by a series of Amazon SageMaker notebooks (which can be find in [this repo] which provide the step-by-step workflow to use very large text datasets for operational purposes, mainly, how to use historical online product customer review data in order to provide predictive capability for identifying product categories, and recommend similar products.

**Use Case Introduction**

The Use Case which will underpin this blog post concerns building a customer support solution, based on customer product reviews [data](https://registry.opendata.aws/amazon-reviews/). Whilst there are many different use cases which can be based on such data, we're going to be focusing on how to use a variety of NLP techniques to help optimize the core functions of an Organisations Customer Support unit. As part of this journey, we're going to navigate this project using a data science framework, which involves data cleansing and preparation, exploration and insights, modelling, scaling, and then operationalization. 

There are several scenarios in which we can explore for this use case, below lists just three which are possible:

 - Incorrect product reviews: customer reviews which are associated with the wrong product can be a time consuming task to track, especially when using manual processes (e.g. Human QA workflows). We can develop automatic processes to predict whether the review added for a specific product (or category of product) is correct, helping reduce some of the human overheads required to support this.
 - Product Categorization - The correct assignment of the product category has several consequences, from technical consequences such as wrongly indexing thus affecting recommendations, to balancing internal product catalogues from a operational perspective. Whilst we can use keywords within the product listing itself to derive the product category, we can also use customer reviews to ensure that the assigned category is suitable, or needs adjusting.
 - Decreasing Product Rating - Reviews could be considered as a good proxy of the sentiment of a product (or category of products), and although we have a star rating with many review systems, the shift in language of a review can help detect if the quality of a product is changing. One technique will be to examine whether the language is shifting overtime.
 
Whilst in this demonstration we're not going to cover all of the use case scenarios, we'll ensure that the solution that we develop is extensible and generalizable to enable more scenarios to be added going forward. 

**Services/Technologies Used**

The use case will use a range of AWS services and technologies, and demonstrate how an organisation can orchastrate different services into a operationalized architecture.


- AWS Glue
- Amazon SageMaker
- Amazon Neptune
- AWS Lambda


**Contents**

- [Data Preparation]() - How to use AWS Glue and SPARK to prepare and process large datasetes ready for analysis
- [Data Experimentation]() - Using Amazon Sagemaker to construct a representative sample of our dataset, and inspect the characteristics of our data. 
- [Model Experimentatio] - Using Amazon SageMaker's built in Algorithms, we'll apply some simple modelling techniques, and then determione which data partitioning / features work best for our tasks.
- [Modelling Scaling]() - Based on the model experiments, we'll scale up the model which yields the most suitable results for our use case
- [Graphing Data]() - Once we've established the correct models to acheive our predictive tasks, we need to find a way to structure our data to address some of the initial asks within the use case defined at the start of the project.
- [Testing Framework]() - One of the critical aspects of introducting a new process within an organisation is to ensure we can test and evaluate our processes when in an operationalized state. This will involve some form of split testing. 
- [Operationalizing]() - Finally, we need to architect our solution in order to ensure we can deploy the solution which can be scaled across an organisation. We'll explore the use of Serverless services such as AWS Lambda and Step Functions to achieve this.



## Data Preparation

Let's first take a look at the data we're going to be using for this Use Case.


### Dataset Details  

> Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazon’s iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Let's just take a look at what the tablular form of the data looks like to get an understanding of what we're working with.


|marketplace|customer_id|review_id     |product_id|product_parent|product_title                                                                        |product_category|star_rating|helpful_votes|total_votes|vine|verified_purchase|review_headline             |review_body|review_date|
|-----------|-----------|--------------|----------|--------------|-------------------------------------------------------------------------------------|----------------|-----------|-------------|-----------|----|-----------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
|US         |22480053   |R28HBXXO1UEVJT|0843952016|34858117      |The Rising                                                                           |Books           |5          |0.0          |0.0        |N   |N                |Great Twist on Zombie Mythos|I've known about this one for a long time, ... |2012-05-03 |
|US         |44244451   |RZKRFS2UUMFFU |031088926X|676347131     |Sticky Faith Teen Curriculum with DVD: 10 Lessons to Nurture Faith Beyond High School|Books           |5          |15.0         |15.0       |N   |Y                |Helpful and Practical       |The student curriculum was better than I expected. The ...|2012-05-03 |
|US         |20357422   |R2WAU9MD9K6JQA|0615268102|763837025     |Black Passenger Yellow Cabs: Of Exile And Excess In Japan                            |Books           |3          |6.0          |8.0        |N   |N                |Paul                        |I found \\"Black Passenger / Yellow Cabs\\" to be a very readable book ... |2012-05-03 |


As we can see from the data table extract above, we have a variety of fields which can be used to filter across, as well as use as features for different analytical insights. 

As with the majority of data science projects which are driven with an underlying business objective, it's important to focus on what information/attributes are goingt to provide the most value and insights to the task at hand. However, selecting such data points, groupings, or segments in the data requires a delicate balance of initial inspection and analysis, hypothesis forming and prooving, as well as a strong level of domain expertise. 

For the purpose of our use case, we're going to initially focus our efforts on several of the attributes in this dataset:

- prodict_cateogry - this represents the higher-order category that a product is assigned to (e.g. Books, Electronics), we will inspect the data to determine how many categories there are (from the dataset description, we can see that they are not extremely granular, which may be useful if we're building multi-class classifiers
- review_body - The main text of the review. This is written by a customer, and describes the customer's feedback and opinion of the product. The review body will be the main feature used in our NLP and classificationt tasks.
- review_date - The date the review was added. This will be useful as we may want to construct temporal views of our data based ont the time it was generated.


### Data Preparation

As we're going to be working with a very large dataset, using the right tool is going to be essential here to view, process, and transform our data to a structure that's suitable for our purpose. We're going to use [AWS Glue](https://aws.amazon.com/glue/) as our environment to process our data, and by doing so, we will be able to use Apache SPARK to process our data. 

For those that are not familar with Apache SPARK, it's simply a large-scale data processing platform/engine, which provides a scalable environment to run very large processing and analytics workloads in. For more details on Spark, see [here](https://spark.apache.org/).

If we navigate to our [Pre-Processing Notebook](), we can see it looks exactly the same as a SageMaker Jupyter Notebook, and that's because it is. The great thing about AWS Glue is that it uses SageMaker Notebooks as the environment to perform your Spark operations; the only difference is that when using AWS Glue, we need to attach a Spark endpoint to a given notebook environment.

Before you can run this notebook, you'll need to have set up your own AWS Glue service via the AWS Console, which will involve configuring your 'Dev Endpoint', and your Notebooks linked to the Dev Endpoint. For more information on setting up your environment, see [here](https://docs.aws.amazon.com/glue/latest/dg/getting-started.html).


**Important Note**: As we're going to be initially working with Spark and AWS Glue, I would recommend creating your Glue Notebook Instance to begin with, and then cloning this repository into your AWS Glue Notebook environment. Once you've completed the Spark processing, it's important to shut down the Dev Endpoint, as this is charged not by usage, but uptime. Your Notebook Instance will still be active after you shutdown your Dev Endpoint, as these are decoupled services.


#### Importing Data and Processing

Our initial workflow is going to involve importing the data from the originating S3 bucket, and then loading it into a PySpark Dataframe so we can process some of the text fields.

```python
reviews = spark.read.parquet("s3://amazon-reviews-pds/parquet")

MIN_SENTENCE_LENGTH_IN_CHARS = 5
MAX_SENTENCE_LENGTH_IN_CHARS = 5000

df = reviews \
  .distinct() \
  .filter("marketplace = 'US'") \
  .withColumn('body_len', F.length('review_body')) \
  .filter(F.col('body_len') > MIN_SENTENCE_LENGTH_IN_CHARS) \
  .filter(F.col('body_len') < MAX_SENTENCE_LENGTH_IN_CHARS) \

record_count = df.count()
print('Total Record Processing: {}'.format(record_count))
```

In the code snippit above, we're simply reading the Amazon Reviews data from Parquet format into our Spark cluster, and then converting it to a PySpark Dataframe, with some simple filters on the length of the length of the review_body text. We do this as we want to ensure that we are working with rows where the reviews are present, and are not the length of several pages. A more robust approach here would be to first select all the reviews, calculate the variance in review_body length, and then use some a suitable range based on some St.d value or orther property.

The output of our processing results in the following:

```sh
Total Record Processing: 148,739,853
```

Depending on the size of the AWS Glue End-point, the processing time will vary. I chose 10 DPUs and it took around 15 minutes.

Next, we're going to process the `review_body` column data, as we want to ensure it is cleaned and ready for use for different NLP tasks. If you've had any experience with text processing for data mining or NLP tasks, then removing terms and tokenizing is typically the first step to readying your data. We're goign to perform some basic stop word removal, stemming, and tokenizing and then add these new filtered reviews to a new column.





## Data Experimentation


to-Do

## Model Experimentation


to-Do


## Scaling Models

to-Do


## Operationalization


to-Do



