import os
import glob

# Search path for modeling_qwen.py under HuggingFace modules
pattern = os.path.expanduser("~/.cache/huggingface/modules/**/modeling_qwen.py")
files = glob.glob(pattern, recursive=True)
print("Found files to patch:", len(files))

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    target1 = "self.rope_theta = config.rope_theta"
    replacement1 = "self.rope_theta = getattr(config, 'rope_theta', 1000000.0)"
    
    target2 = "past_key_values.get_usable_length(seq_length)"
    replacement2 = "(past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values.get_usable_length(seq_length))"
    
    target3 = "past_key_value.get_usable_length(kv_seq_len, self.layer_idx)"
    replacement3 = "(past_key_value.get_seq_length(self.layer_idx) if hasattr(past_key_value, 'get_seq_length') else past_key_value.get_usable_length(kv_seq_len, self.layer_idx))"
    
    modified = False
    if target1 in content:
        content = content.replace(target1, replacement1)
        modified = True
    if target2 in content:
        content = content.replace(target2, replacement2)
        modified = True
    if target3 in content:
        content = content.replace(target3, replacement3)
        modified = True
        
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully patched: {file_path}")
    else:
        print(f"No targets found in: {file_path}")
