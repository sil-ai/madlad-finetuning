import argparse
from clearml import Task
from pathlib import Path
from utils.ingest_zip import convert_paratext_to_vref
from utils.upload_data import upload_data



def preprocess_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    # If file is a zip file, convert it to vrefs
    if file_path.suffix == ".zip":
        file = convert_paratext_to_vref(file_path.read_bytes())
    else:
        file = file_path.read_bytes()
    with open(f'./data/{file_path.stem}.txt', 'wb') as f:
        f.write(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Path to the source file")
    parser.add_argument("--target", type=str, help="Path to the target file")
    parser.add_argument("--source-lang", type=str, help="Source language ISO code")
    parser.add_argument("--target-lang", type=str, help="Target language ISO code")
    args = parser.parse_args()

    print(args.source)
    print(args.target)

    source_file_path = Path(args.source)
    target_file_path = Path(args.target)

    preprocess_file(source_file_path)
    preprocess_file(target_file_path)

    upload_data()

    task = Task.create(
        project_name="IDX Translation Fine-tuning/IDX MADLAD Exp",
        task_name=f"madlad-finetuning {source_file_path.stem.split('-')[0]}-{target_file_path.stem.split('-')[0]}",
        script="train.py",
        argparse_args=[('source', args.source), ('target', args.target), ('source-lang', args.source_lang), ('target-lang', args.target_lang)],
        add_task_init_call=True,
        requirements_file="./requirements.txt",
        docker="python:3.8.10",
    )


    task.execute_remotely(queue_name='jobs_backlog')

    
    
        