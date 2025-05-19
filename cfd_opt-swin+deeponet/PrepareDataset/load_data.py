import csv
# Load the CSV files of the training set and the validation set
def load_case_ids(csv_file):
    ids = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['case_id']))
    return ids