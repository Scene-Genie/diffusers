from datasets import load_dataset

# # Load the dataset
# dataset = load_dataset('scene-genie/train-reflections-shadows')

# # Filter the dataset
# filter_ids = open('filter_ids.txt').read().splitlines()

# filtered_dataset = dataset.filter(lambda x: x['image_id'] not in filter_ids)
# filtered_dataset.push_to_hub("scene-genie/train-reflections-shadows-filtered")

ds = load_dataset('scene-genie/train-reflections-shadows-filtered', split = "train")
shadow_ds = ds.filter(lambda x: 'shadow' in x["prompt"])

shadow_ds.push_to_hub("scene-genie/train-shadows-filtered")