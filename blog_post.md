# Practical Guide: Using Product Reviews to Support Internal Customer Operations


## Overview

In the following blog post, we're going to explore how to use AWS to scale data science experiments, with a particular focus on building Text Classifiers for supporting different functions within an organisation.

This blog post is supported by a series of Amazon SageMaker notebooks (which can be find in [this repo] which provide the step-by-step workflow to use very large text datasets for operational purposes, mainly, how to use historical online product customer review data in order to provide predictive capability for identifying product categories, and recommend similar products.

**Use Case Introduction**

The Use Case which will underpin this blog post concerns building a customer support solution, based on customer product reviews [data](https://registry.opendata.aws/amazon-reviews/). Whilst there are many different use cases which can be based on such data, we're going to be focusing on how to use a variety of NLP techniques to help optimize the core functions of an Organisations Customer Support unit. As part of this journey, we're going to navigate this project using a data science framework, which involves data cleansing and preparation, exploration and insights, modelling, scaling, and then operationalization. 

There are several scenarios in which we can explore for this use case, below lists just three which are possible:

 - Incorrect product reviews: customer reviews which are associated with the wrong product can be a time consuming task to track, especially when using manual processes (e.g. Human QA workflows). We can develop automatic processes to predict whether the review added for a specific product (or category of product) is correct, helping reduce some of the human overheads required to support this.
 - Product Categorization - The correct assignment of the product category has several consequences, from technical consequences such as wrongly indexing thus affecting recommendations, to balancing internal product catalogues from a operational perspective. Whilst we can use keywords within the product listing itself to derive the product category, we can also use customer reviews to ensure that the assigned category is suitable, or needs adjusting.
 - Decreasing Product Rating - Reviews could be considered as a good proxy of the sentiment of a product (or category of products), thus the language of a review can help detect if the quality of a product is changing. 
 

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
- [Graphing Data]() - Once we've established the correct models to acheive our data


