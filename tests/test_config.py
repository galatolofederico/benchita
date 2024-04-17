CONFIG = """experiment: cerbero-7b

tasks:
  - name: ironita
    num_shots: 3
  - name: sentipolc
    num_shots: 3

models:
  - model:
      name: galatolo/cerbero-7b
      class: AutoModelForCausalLM
      args:
        attn_implementation: flash_attention_2
        dtype: float16
    template:
      system_style: inject
    generate:
      batch_size: 16
      args:
        do_sample: False
"""

import tempfile
from benchita.config import parse_config

def test_config():
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(CONFIG)
        f.seek(0)
        
        config = parse_config(f.name)
        print(config)