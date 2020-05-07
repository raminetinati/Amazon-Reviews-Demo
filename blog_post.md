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


> Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazon’s iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Let's just take a look at what the tablular form of the data looks like to get an understanding of what we're working with.


|marketplace|customer_id|review_id     |product_id|product_parent|product_title                                                                        |product_category|star_rating|helpful_votes|total_votes|vine|verified_purchase|review_headline             |review_body                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |review_date|
|-----------|-----------|--------------|----------|--------------|-------------------------------------------------------------------------------------|----------------|-----------|-------------|-----------|----|-----------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
|US         |22480053   |R28HBXXO1UEVJT|0843952016|34858117      |The Rising                                                                           |Books           |5          |0.0          |0.0        |N   |N                |Great Twist on Zombie Mythos|I've known about this one for a long time, but just finally got around to reading it for the first time.  I enjoyed it a lot!  What I liked the most was how it took a tired premise and breathed new life into it by creating an entirely new twist on the zombie mythos.  A definite must read!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |2012-05-03 |
|US         |44244451   |RZKRFS2UUMFFU |031088926X|676347131     |Sticky Faith Teen Curriculum with DVD: 10 Lessons to Nurture Faith Beyond High School|Books           |5          |15.0         |15.0       |N   |Y                |Helpful and Practical       |The student curriculum was better than I expected. The lessons were easy to understand and easy to teach. Each lesson has a list up front explaining everything you need for the session. This alone is very helpful.<br /><br />As for the lessons, the students in my Sunday School class (11th and 12th graders) are being challenged by new ideas and thoughts. This has led them to really look at their faith and figure out what they truly believe. This alone is essential to them establishing a personal faith and not cling to the faith of their parents.<br /><br />This curriculum is essential to any youth group. They are topics that should be discussed with every senior before they leave high school.<br /><br />How could you pass up on such a formational curriculum for such a great price?<br /><br />- Zach Carpenter<br />Roseburg Church On The Rise|2012-05-03 |
|US         |20357422   |R2WAU9MD9K6JQA|0615268102|763837025     |Black Passenger Yellow Cabs: Of Exile And Excess In Japan                            |Books           |3          |6.0          |8.0        |N   |N                |Paul                        |I found \\"Black Passenger / Yellow Cabs\\" to be a very readable book.  It was fun going hearing about Mr. Bryan's sexual escapades -- at first.  It was refreshing to have someone who could frequently be very perceptive speak honestly not only about his life, but the sexual escapades and shenanigans he was engaging in.  And to link that behavior with the horrific conditions of his early youth in a ghetto in Jamaica, and more loosely, with the Jamaican cultural socialization which influenced him.  These elements humanized and explained some of his less than noble behavior.<br /><br />However, what becomes extremely obvious as the book goes on (and on and on it does go)is that, even in Mr. Bryan's own words, he suffers from 'dissocial personality disorder,' or some such thing.  Perilously close, it seems to me, to sociopathic behavior.  Even the title, in retrospect, speaks to the sociopaths view of humanity.  Mr. Bryan is a 'passenger', i.e. a person, the woman are 'cabs' -- non-humans.<br /><br />The irony, of course, for anyone who's read the novel is that Mr. Bryan frequently speaks of how the Japanese culture traumatizes and demeans women.  However, he is a flagrant abuser of women -- though he frequently presents himself as the opposite, as one who has nutured these poor women back to health!<br /><br />He has impregnated more than 10 women while in Japan -- and in most case paid or helped to pay for the abortions.  More than 10 women!  All because he enjoys the feel of sex without a condom.  And because he knows that most of the women won't insist that he wear one due to their timidity!  So, in effect, he'd have woman after woman go through the greater trauma of an abortion; and have fetus be terminated (some would even say murdered); and he would even have to pay (in some cases only half) for his temporary pleasure!  This, more than ten times!<br /><br />And let us not forget how he then has his cake and eats it too as he makes out how horrible it is of the Japanese to run an abortion industry, instead of having oral contraceptives more readily available.<br /><br />He also even refers to the women in his writings as cars.  He prefers, he tells the reader, his \\"bentley\\" (Shoko) but the 'ferrari' (Azusa) is sexier and her parents are so accepting of him!<br /><br />The language he uses time and again, in conjunction with the repeated behaviors, indicates that closeness he shares with the sociopath: people are things to be used, manipulated.  People are not people: they are not truly human with feelings and the right to be treated as 'ends in themselves' -- but rather they are trappings, are thigns, to be used and manipulated by the sociopath.<br /><br />Mr. Bryan frequently talks about how he doesn't really love -- indeed, it seems that he doesn't even like very much -- Azusa.  But she's so good looking that he wants to have children with her; while, of course, marrying Shoko (or perhaps converting to Islam -- though he's an atheist -- in the hopes of being able to legally marry both of them).<br /><br />This goes on and on.<br /><br />Again, at first, much of this seemed fun, interesting, a man with hardbreaks coming into his own.  However, this behavior is -- and has been -- going on for years, even into his forites.<br /><br />Yes, there's the 'addiction,' however, I got the distinct impression that Mr. Bryan was being more honest than perhaps he meant to be when he on repeated occasions told the reader about his personality disorder.  He truly has one, it seems.  He may not be a sociopath, but in many respects his 'disorder' is such that other people are things.  For me, to get behind the eyes, so to speak, of a person like that is a frightening experience.  But, pun intended, an 'eye-opener'!<br /><br />It's also curious that Mr. Bryan has, it seems, only female friends.  In part, he would have the reader believe it is because of his greater sensitivity to females and their plight.  He sometimes refers to himself as a 'lesbian in a male's body.'  His sensitivity is hogwash!  Woman form the perfect dovetail for the likes of him:  one, he likes 'tail' and two, he knows that he can manipulate them far more easily than he can men.  These two features make them delectable to him.<br /><br />A minor note.  Much of the book is fairly well written with this curious feature:  there are numerous rather 'lofty' words sprinkled throughout the text.  For example he writes of one woman \\"she tried corybanticly to cloak herself with anything in sight\\" (Yukari; p331).  Now, I have a master's degree in English, but I don't know what the freak \\"corybanticly\\" means.  The point is that too frequently this text is larded with extremely high falutin words, amid, of course, the booty calls, and his petrified rod, etc.<br /><br />Most curious!<br /><br />Anyway, this is in many ways an interesting read: some of the incidents are enjoyable to read; much of the writing is actually pretty good; there are some sharp insights into Japanese culture ( I lived in Japan for a year so I had some access to what he's referring to); there's great awareness generally of culture and its influence (Jamaican, American, Japanese).  There's the fascinating other perspective -- both replusive and fascinating -- of the view from the 'personality disorder'.<br /><br />Anyway, I give this book 3 stars.<br /><br />I hope this review helped.<br /><br />Aloha,<br />paul|2012-05-03 |



### Dataset Details



## Data Experimentation


to-Do

## Model Experimentation


to-Do


## Scaling Models

to-Do


## Operationalization


to-Do



