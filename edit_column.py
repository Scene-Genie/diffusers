from datasets import load_dataset

ds = load_dataset("scene-genie/instagram-dataset-train")
ds = ds.map(lambda x: {"prompt": "add reflection to the can"})

ds.push_to_hub("scene-genie/instagram-dataset-train")