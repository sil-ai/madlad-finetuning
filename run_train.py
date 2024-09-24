from clearml import Task

task = Task.create(
    project_name="IDX Translation Fine-tuning/IDX MADLAD Exp",
    task_name="madlad-finetuning",
    script="train.py",
    add_task_init_call=True,
    requirements_file="./requirements.txt",
    docker="python:3.8.10",
)


task.execute_remotely(queue_name='jobs_backlog', exit_process=True)