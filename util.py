import csv
import os
import shutil

def split_csv_by_rows_stream(input_filename, num_chunks, dir_path):
    """Splits a large CSV file into smaller chunks by rows without loading the whole file into memory."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(input_filename, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    chunk_size = total_rows // num_chunks
    if total_rows % num_chunks != 0:
        chunk_size += 1
    
    with open(input_filename, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for i in range(num_chunks):
            chunk_filename = f"{dir_path}/chunk_{i}.csv"
            
            with open(chunk_filename, mode="w", newline="", encoding="utf-8") as chunk_file:
                writer = csv.writer(chunk_file)
                writer.writerow(header)
                
                for _ in range(chunk_size):
                    try:
                        row = next(reader)
                        writer.writerow(row)
                    except StopIteration:
                        break
            
            print(f"Chunk {i + 1} saved to {chunk_filename}")