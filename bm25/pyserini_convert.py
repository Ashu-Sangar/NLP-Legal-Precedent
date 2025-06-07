# Preprocessing: Convert Caselaw JSON files (PA State Volumes 1-50) to Pyserini Index Format

import os
import json

output_dir = "indexed_case_corpus_vol_1-50_for_pyserini"
os.makedirs(output_dir, exist_ok = True)

# Loop over unzipped bulk data download from Caselaw and convert
# to pyserini json format to prepare for document indexing.
# {
#     "id": "doc1",
#     "contents": "this is the contents."
# }
for volume in range (1, 51): # Only loops over first 50 volumes of PA state data
    volume_path = os.path.join("..", "data", "Caselaw_Pennsylvania_State_Reports_1845-2017", str(volume), "json")

    if not os.path.isdir(volume_path):
        continue

    for file in os.listdir(volume_path):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(volume_path, file)

        with open(file_path, "r", encoding = "utf-8") as f:
            try:
                data = json.load(f)
                doc_id = str(data["id"]) # Use unique id from original data as the doc_id

                # The new "contents" will be the former "opinions" section, 
                # if there are no opinions files will be skipped.
                # (will figure out something better later)
                opinions = data.get("casebody", {}).get("opinions", [])
                if not opinions or "text" not in opinions[0]:
                    print(f"Skipping {file_path} due to missing opinions.")
                    continue
                contents = opinions[0]["text"].strip()

                converted = {"id": doc_id, "contents": contents}

                output_path = os.path.join(output_dir, f"{doc_id}.json")

                with open(output_path, "w", encoding = "utf-8") as out_f:
                    json.dump(converted, out_f)
            
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")