from datasets import load_dataset

def get_training_and_validation_datasets(data_file):
    try:
        # Load the dataset
        dataset = load_dataset("json", data_files=data_file)
        print("Dataset loaded successfully")

        # Split the dataset into training and validation sets
        train_test_split = dataset['train'].train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        print("Dataset split into training and validation sets")

        return train_dataset, eval_dataset
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
