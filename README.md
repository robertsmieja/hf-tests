HuggingFace Tests
===

A loose collection of Python scripts to play around with HuggingFace libraries

# Requirements
 - [Python 3.9 - 3.12 ](https://www.python.org/downloads/)
   - This is due to [PyTorch](https://pytorch.org/get-started/locally/)
 - [Poetry](https://python-poetry.org/docs/#installing-with-pipx)
    - It's recommended to use [pipx](https://pipx.pypa.io/stable/installation/) to install Poetry

# Instructions
```shell
cd ~/git/hf-tests

# setup virtual environment and install dependencies
poetry install

# enter a shell inside the venv
poetry shell

# run tests
python ./test.py
```

## Example output (using CPU)
```shell
python .\test.py
```
```shell
<torch.utils.benchmark.utils.common.Measurement object at 0x000002455A3304D0>
classifier(sample_input)
  22.37 ms
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x00000245604CA1D0>
generator(sample_input, pad_token_id=generator.tokenizer.eos_token_id)
  1.31 s
  1 measurement, 10 runs , 1 thread
```
