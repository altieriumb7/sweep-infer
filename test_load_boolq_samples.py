from datasets import load_dataset

ds = load_dataset('google/boolq', split='validation')
target_indices = [8, 94, 67, 88, 39, 50, 43, 61, 73, 0]
print(f"Loaded {len(ds)} items. Inspecting 10 target samples:")
for i, idx in enumerate(target_indices):
    item = ds[idx]
    print(f"{i+1:2d}. idx={idx:4d} | target={str(item['answer']):<5} | Q: {item['question']}")
