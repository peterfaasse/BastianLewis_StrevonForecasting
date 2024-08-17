import os
import re
from datetime import datetime

def get_most_recent_file(folder_path: str) -> str:
    """
    Finds the most recent file in a folder based on the date in the filename.
    The date is assumed to be in the format 'yyyymmdd'.

    Parameters:
    -----------
    folder_path : str
        The path to the folder containing the files.

    Returns:
    --------
    str
        The full path of the most recent file based on the date in the filename.
        Returns None if no files with a date in the name are found.
    """
    date_pattern = re.compile(r'(\d{8})')  # Regex to match 'yyyymmdd' format
    most_recent_file = None
    most_recent_date = None

    for filename in os.listdir(folder_path):
        match = date_pattern.search(filename)
        if match:
            file_date = datetime.strptime(match.group(0), '%Y%m%d').date()
            if most_recent_date is None or file_date > most_recent_date:
                most_recent_date = file_date
                most_recent_file = filename

    if most_recent_file:
        return os.path.join(folder_path, most_recent_file)
    else:
        return None

# Example usage:
# folder_path = '/path/to/your/folder'
# most_recent_file = get_most_recent_file(folder_path)
# print(f"Most recent file: {most_recent_file}")
