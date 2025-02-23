import kagglehub
import shutil
import os
from pathlib import Path

class IncorrectCurrentDirectoryError(Exception):
    """Error for not running the function in the project root."""
    def __init__(self, message):
        super().__init__(message)

def check_dataset(
        filename: str='Extended_Employee_Performance_and_Productivity_Data.csv'
        ) -> bool:
    "Check if the dataset needs to be downloaded."
    cwd = Path.cwd()
    dataset_path = os.path.join(cwd, 'dataset', filename)
    return os.path.exists(dataset_path)

def check_cached_dataset(dataset_endpoint: str, filetype: str):
    "Check if the dataset is already cached and need to be downloaded."
    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets"
    cache_path = (cache_base / dataset_endpoint.replace("/", "\\")
                  / "versions" / "1")
    files = list(cache_path.glob(f"*{filetype}"))
    return (cache_path.exists() and any(cache_path.iterdir()),
            files)
        

def download_data_from_kaggle(dataset_endpoint: str) -> str:
    """Download data from Kaggle only if it's not already in the cache."""
    cache_exists, cache_path = check_cached_dataset(dataset_endpoint,
                                                    filetype='.csv')
    if cache_exists:
        print(f"Dataset already exists in Kaggle cache: {str(cache_path[0])}")
        return str(cache_path[0])
    path = kagglehub.dataset_download(dataset_endpoint)
    print("Downloaded dataset to:", path)
    return path

def move_dataset_to_project_folder(source: str,
                                   destination: str) -> None:
    "Move downloaded data to this project's folder."
    cwd = Path.cwd()
    if (not (cwd / "notebooks").exists()
        and (cwd / "conda_packages.txt").exists()):
        raise IncorrectCurrentDirectoryError(
            "Please execute the file in the project root:"
            f"Current Directory: {cwd}"
        )
    os.makedirs(destination, exist_ok=True)
    shutil.move(source, destination)
    print(f"Dataset moved to: {destination}")

def download_protocol(
        dataset_endpoint="mexwell/employee-performance-and-productivity-data"
        ):
    dataset_saved = check_dataset()
    destination = os.path.join(Path.cwd(), 'dataset/')
    if not dataset_saved:
        path = download_data_from_kaggle(dataset_endpoint=dataset_endpoint)
        move_dataset_to_project_folder(path, destination=destination)
    else:
        print('Dataset already ok.')

if __name__=='__main__':
    dataset_endpoint = "mexwell/employee-performance-and-productivity-data"
    download_protocol(dataset_endpoint=dataset_endpoint)
    # print(check_cached_dataset(dataset_endpoint=dataset_endpoint,
    #                            filetype='.csv'))
