# Pontifficate

A tool for analysis of paired DAPI and FITC fluorescence microscopy images. Pontifficate creates cell segmentations from DAPI images and measures intensity patterns in corresponding FITC images.

## Features

- Cell segmentation using DAPI nuclear staining
- Detection and counting of fluorescent foci in FITC images
- Measurement of normalized intensity within cell boundaries
- Advanced segmentation options including watershed algorithm and edge detection
- Individual CSV reports for each image pair with per-cell measurements
- Visualizations of segmentation steps and foci detection

## Project Structure
pontifficate/
│
├── LICENSE                
├── README.md               
├── pyproject.toml          
├── tests/                    
│   ├── __init__.py          
│   ├── test_segmentation.py  # Tests for segmentation functions
│   ├── test_normalization.py # Tests for image normalization
│   ├── integration/          # Integration tests
│   │   └── test_paired_analysis.py # Tests for paired analysis pipeline
│   └── ...                 
│
├── pontifficate/              
│   ├── __init__.py         
│   ├── utils.py              # Helper functions for file I/O
│   ├── normalization.py      # Image normalization functions
│   ├── segmentation.py       # Cell segmentation with watershed and edge detection
│   ├── paired_analysis.py    # Main functions for DAPI/FITC paired analysis
│   ├── cli.py                # Command-line interface
│   ├── logging_config.py     # Logging configuration
│   └── metadata.py           # Metadata extraction and processing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pontifficate.git
   cd pontifficate
   ```

2. Install with Poetry:
   ```
   poetry install
   ```

3. Or install with pip:
   ```
   pip install -e .
   ```

## Usage

### Paired DAPI/FITC Analysis (Main Functionality)

The primary function of Pontifficate is to analyze paired DAPI and FITC images:

```bash
pontifficate analyze-pairs INPUT_DIR OUTPUT_DIR [OPTIONS]
```

Where:
- `INPUT_DIR` is a directory containing paired DAPI and FITC images
- `OUTPUT_DIR` is where results will be saved

Input images should follow a naming convention:
- DAPI images: `sample_name_dapi.tiff`
- FITC images: `sample_name_fitc.tiff`

For each image pair, the tool will:
1. Create cell segmentations from the DAPI image
2. Detect and count fluorescent foci in the FITC image
3. Measure normalized intensity within each cell
4. Generate a CSV file with per-cell measurements
5. Create visualizations of the segmentation and foci detection

#### Options

- `--watershed/--no-watershed`: Enable/disable watershed segmentation (default: enabled)
- `--edge-detection/--no-edge-detection`: Enable/disable edge detection (default: enabled)
- `--min-cell-area INTEGER`: Minimum area in pixels for a region to be considered a cell (default: 200)
- `--combined-csv PATH`: Optional path to save a combined CSV with data from all samples

Example:
```bash
pontifficate analyze-pairs data/samples/ results/ --min-cell-area 150 --no-watershed
```

### Image Normalization

The package also supports basic image normalization:

```bash
pontifficate normalize INPUT_PATH OUTPUT_DIR
```

Where:
- `INPUT_PATH` can be a single image or directory of images
- `OUTPUT_DIR` is where normalized images will be saved

## Output Files

For each image pair, the following files are generated:

- `sample_name_cell_analysis_results.csv`: CSV file with per-cell measurements including:
  - Cell ID
  - Foci count
  - Normalized intensity
  - Cell area

- `sample_name_visualization.png`: Visualization with:
  - DAPI image
  - FITC image
  - Cell segmentation with ID labels
  - Detected foci

- `sample_name_segmentation_steps.png`: Visualization of segmentation process:
  - Original image
  - Basic thresholding
  - Edge detection (if enabled)
  - Watershed segmentation (if enabled)

## Code Formatting and Linting

For linting, run:
```sh
poetry run pylint ./pontifficate ./tests
```

For running tests:
```sh
poetry run pytest
```