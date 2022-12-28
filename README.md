# AWS Pipeline for Predictive Analysis

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/cover01.png" width="875" height="400" />

## Introduction

By the end of this project we would have costructed a data pipeline that 
orchestrates a batch ETL process in Amazon EMR and build a predictive estimate 
within our analysis for the categorical variable.

Our business requirement for the machine learning engineer is to extract, transform, 
and load the data in such a way that our model can be trained using a clean input. 
Before we jump onto data analytics, we must immediately address the need to 
update this procedure whenever new data is introduced. However, in order to utilize the output 
result to train our machine learning model, we first need to build a pipeline that summarizes 
the average income of each desired dimension from the dataset. 

## Architecture Overview



<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/architecture_design01.JPG" width="950" height="170" />

The following resource set-up is required for our Amazon MWAA environment:

* Each availability zone has an S3 gateway VPC endpoint to provide a secure connection between Amazon MWAA and Amazon S3.
* Each Availability Zone that the VPC spans has a firewall subnet and a customer subnet and that the VPC has an internet gateway for internet access.
* Each availability zone has an EMR interface VPC port to provide a safe connection between AWS MWAA and Amazon EMR.

## Requirements

`f_training_data.csv`, `f_testing_data.csv`, `target_training_data.csv` files make up the data source, which is kept in an Amazon S3 bucket in the cloud. 
Each observation represents a specific job listing, and each column contains particular details
about the applicant and the position they applied for.


<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/sample_data_img3.png" width="805" height="295" />


## Understanding the ETL Pipeline Design 

We can break-down the orchestration of the pipeline by looking at how each step interacts
within the data life cycle: 

* we begin by **triggering the DAG** 
* afterwhich we are able to initialize the **EMR cluster** and 
* submit a spark step in the **EMR cluster nodes** that execute the **ETL workflow** 
* after this step is complete we need to **terminate EMR cluster** and finally **end the DAG** process


<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/dag_trigger_workflow.png" width="900" height="90" />


## Creating an EC2 Instance Keypair

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/emr_ec2_defaultrole.JPG" width="875" height="700" />



## Creating an S3 Bucket

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/Creating the S3 Bucket_.JPG" width="850" height="210" />


## Initializing MWAA airflow Environment in CloudFormation

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/cloudformation_mwaa_env.JPG" width="900" height="320" />

## Running DAG on Apache Airflow

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/airflow_design.jpg" width="850" height="250" />

## Exploratory Analysis

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/exploratory_step.png" width="955" height="1225" />

## Correlation Matrix

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/correlation_matrix.png" width="955" height="655" />


## Employee Salary Distribution

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/employee_distribution.png" width="950" height="305" />

## Employee Education vs Salary

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/avg_salary_major.png" width="950" height="390" />

## Linear Regression Performance

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/Linear_Regression_Performance3.JPG" width="955" height="550" />

## Gradient Boosting Regressor

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/GBR_Performance3.JPG" width="955" height="650" />


## Conclusion Based on Model Performance 

Even though both of the models delivered a promising estimate for the predicted values, we notice that Gradient Boosting Regressor has the lowest mean square error. Hence, GBR is the most accurate model to predict the desired attribute.

The evaluated features of impact for the Gradient Boosting Regressor are as follows:

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/feauture_significance01.jpg" width="920" height="425" />

Out of all the criteria, years of experience, distance from the city, and the type of work have the most bearing on income prediction. Whereas the school's major carries the least value, in contrast.

## And at Last, the Gradient Boosting Regressor Prediction

<img src="https://github.com/JoexTitan/AWS-Pipeline-Predictive-Analysis/blob/master/visuals/gbr_salary_prediction.png" width="850" height="280">






