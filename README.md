
## Project Structure
pontifficate/
│
├── LICENSE                
├── README.md               
├── pyproject.toml           
├── tests/                    
│   ├── __init__.py          
│   ├── test_mask_creation.py # Tests for mask creation
│   ├── test_normalization.py # Tests for image normalization
│   └── ...                 
│
├── pontifficate/               
│   ├── __init__.py         
│   ├── utils.py              # Fun helper functions
│   ├── normalization.py      # Image normalization functions
│   ├── segmentation.py       # Cell segmentation and mask creation
│   ├── analysis.py           # Functions to analyze fluorescence intensity
│   ├── visualization.py      # Functions for boundary overlays and visual QC
│   └── metadata.py           # Metadata extraction and processing
│
├── examples/               
│   ├── process_single_image.py
│   └── batch_processing.py

## Installation

1. Clone git repo into directory of choice
2. Open main repo directory
3. run `poetry install`
4. Run your module of choice with `poetry run {module}`

## Modules


## Code Formatting and Linting Standards

For linting, simply run [pylint](https://pylint.pycqa.org/en/latest/) from the root folder:
```sh
poetry run pylint ./image_normalization ./tests
