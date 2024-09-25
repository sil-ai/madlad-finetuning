# create example dataset
from clearml import Dataset

# Create a dataset with ClearML's Dataset class

def upload_data():
    
    current_dataset = Dataset.get(dataset_name="Vref Files", dataset_project="IDX Bible Data")

    dataset = Dataset.create(
        dataset_project="IDX Bible Data",
        dataset_name= "Vref Files",
        # output_uri="s3://clearml-madlad-finetuning-data",
        parent_datasets=[current_dataset]
    )

    # dataset = Dataset.get(dataset_id="1fb5892063ae4b18b9724cd9ef931c06")

    # Add the example csv
    dataset.add_files(path="data/")

    # Upload dataset to ClearML server (customizable)
    dataset.upload()

    # Commit dataset changes
    dataset.finalize()
