from airflow.utils.dates import days_ago
from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.emr_terminate_job_flow_operator import EmrTerminateJobFlowOperator
from airflow.contrib.operators.emr_create_job_flow_operator import *
from airflow.contrib.operators.emr_add_steps_operator import *
from airflow.contrib.sensors.emr_step_sensor import *

DEFAULT_ARGS = {
    'owner': 'airflow',
    'email': ['JoexTitan@gmail.com'],
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

JOB_FLOW_OVERRIDES = {
    'Name': 'slr_expectation_emr',
    'ReleaseLabel': 'emr-6.3.0',
    'Applications': [
        {"Name": 'Hadoop'},
        {'Name': 'Spark'}
    ],    
    'Instances': {
        'InstanceGroups': [
            {
                'Name': "Master node",
                'Market': 'SPOT',
                'InstanceRole': 'MASTER',
                'InstanceType': 'm3.xlarge',
                'InstanceCount': 1,
            },
            {
                'Name': "Worker nodes",
                'Market': 'SPOT',
                'InstanceRole': 'CORE',
                'InstanceType': 'm3.xlarge',
                'InstanceCount': 2,
            }
        ],
        'KeepJobFlowAliveWhenNoSteps': True,
        'TerminationProtected': False,
        'Ec2KeyName': 'NV-keypair',
    },
    'VisibleToAllUsers': True,
    'JobFlowRole': 'EMR_EC2_DefaultRole',
    'ServiceRole': 'EMR_DefaultRole',
    'LogUri': 's3://destination_path'
}

SPARK_STEPS = [
    {
        'Name': 'find_avg_salary',
        'ActionOnFailure': 'CANCEL_AND_WAIT',
        'HadoopJarStep': { 
            'Jar': 'command-runner.jar',
            'Args': ['spark-submit', 
                     '--deploy-mode', 'client', 
                     's3://destination_path'], 
        }
    }
]

with DAG(
    dag_id='DAG_for_avg_slr',
    description='AAFlow orchestrates the workflow in our EMR cluster',
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval='0 0 0 * *',
    tags=['pip_slr']
) as dag:

    begin = DummyOperator(
        task_id='begin_workflow'
    )

    create_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides=JOB_FLOW_OVERRIDES,
        aws_conn_id='aws_default',
        emr_conn_id='emr_default',
    )

    add_step = EmrAddStepsOperator(
        task_id='submit_spark_application',
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
        aws_conn_id='aws_default',
        steps=SPARK_STEPS,
    )

    check_step = EmrStepSensor(
        task_id='check_submission_status',
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster', key='return_value') }}",
        step_id="{{ task_instance.xcom_pull(task_ids='submit_spark_application', key='return_value')[0] }}",
        aws_conn_id='aws_default',
    )

    remove_cluster = EmrTerminateJobFlowOperator(
        task_id='terminate_emr_cluster',
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
        aws_conn_id='aws_default',
    )

    end = DummyOperator(
        task_id='end_workflow'
    )

    begin >> create_cluster >> add_step >> check_step >> remove_cluster >> end



