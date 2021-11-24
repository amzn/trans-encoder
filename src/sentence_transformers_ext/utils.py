# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os 
import csv

def write_csv_log(output_path, csv_file, csv_headers, things_to_write):
    """
    Write logs to a csv file.
    Parameters
    ----------
        output_path: a string specifying the write path
        csv_file: a string specifying the csv file name
        things_to_write: a list of numbers to be written
    Returns
    ----------
        None
    """
    csv_path = os.path.join(output_path, csv_file)
    output_file_exists = os.path.isfile(csv_path)
    with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not output_file_exists:
            writer.writerow(csv_headers)

        writer.writerow(things_to_write)