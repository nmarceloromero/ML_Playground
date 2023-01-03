# ML_Playground

This repository contains Python notebooks with small Machine Learning projects. Publicly available datasets are used in general.

## 1. Water Quality Prediction

The **Water Quality** notebook presents a task in which the objective is to predict if the water is drinkable or not, considering some features. First, some exploratory data analysis is done, and then some classical Machine Learning are used to perform the task.

## 2. Disaster Tweets Classification

The **Disaster Tweets BERT MLP** notebook presents a task to perform classification of tweets (we need to define if the tweet is about a natural disaster or not). For this purpose, the BERT pre-trained network is used as feature extractor, and an MLP is trained and used as the classifier. 

## 3. Amazon Reviews
The **Amazon Reviews Unsupervised** notebook is about processing a sub-set of the Amazon Polarity Dataset. The task proposed by the creators of the dataset is to classify reviews into positive or negative. We will focus more concretely on unsupervised methods to treat the available data.

## 4. Streamlit App
The **Streamlit_App** folder contains a small project to deploy a simple Question-Answering application for two languages (French and Spanish) using Docker. It uses pre-trained models from the [transformers](https://huggingface.co/docs/transformers/index) library. If you want to try it, follow these instructions : 

Navigate to the **Streamlit_App** folder and build a Docker image with the tag "streamlitapp": 

<code>$ docker build -t streamlitapp .</code>

Run the container using: 

<code>$ docker run -p 8501:8501 streamlitapp</code>

The app is now available through the URL : <code>http://0.0.0.0:8501</code>
