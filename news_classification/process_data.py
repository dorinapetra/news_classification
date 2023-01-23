from datetime import datetime

import click
import pandas as pd
from datasets import DatasetDict


@click.command()
@click.argument('data_path')
@click.argument('target_path')
def main(data_path, target_path):
    dataset = DatasetDict.load_from_disk(data_path)

    dataset = dataset.filter(lambda x: x["date_of_creation"] > datetime(2003, 1, 1))
    dataset = dataset.filter(lambda x: x["date_of_creation"] < datetime(2023, 1, 1))
    dataset = dataset.filter(lambda x: x["domain"] != "telex.hu")
    dataset = dataset.filter(lambda x: x["domain"] != "metropol.hu")

    train_df = dataset['train'].to_pandas()
    dev_df = dataset['validation'].to_pandas()
    test_df = dataset['test'].to_pandas()

    data_df = pd.concat([train_df, dev_df, test_df])

    label_count = data_df.groupby('label').count()['uuid']

    with open(target_path) as f:
        f.write(str(label_count))




if __name__ == '__main__':
    main()