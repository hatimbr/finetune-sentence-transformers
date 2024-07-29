# finetune-sentence-transformers

## Usage
Example of usage:
```bash
python -m pytrain
```

Arguments can be passed by command line or by a configuration file or both (command line arguments will overwrite configuration file arguments).
Here is an exemple:
```bash
python -m pytrain --config_file config.ini --batch_size 8 --dev_test
```

Check pytrain/config.py for the list of arguments.