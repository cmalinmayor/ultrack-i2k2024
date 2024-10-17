# Ultrack I2K 2024 - Multiple hypotheses tracking

This tutorial shows Ultrack's multiple hypotheses tracking capabilities. 

Here, rather than searching for an optimal segmentation parameter, we sampled multiple segmentations with different parametrizations and used Ultrack to find the best segments, obtaining more accurate cell tracking.


## Download Dataset

Download the Fluo-C2DL-Huh7 dataset from the [Cell Tracking Challenge](celltrackingchallenge.net), which contains fluorescence microscopy images for cell tracking.

The dataset will be used for demonstrating the segmentation and tracking workflow.


```python
!wget -nc http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip
!unzip -n Fluo-C2DL-Huh7.zip
```

## Import Libraries

Import the libraries needed for reading images, processing them, cell segmentation, tracking, and performance metrics. 


```python
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import napari

from ctc_metrics import evaluate_sequence
from dask.array.image import imread
from napari.utils import nbscreenshot
from numpy.typing import ArrayLike
from rich import print

from ultrack import Tracker, MainConfig
from ultrack.imgproc import normalize, Cellpose
from ultrack.utils.array import array_apply
from ultrack.utils import labels_to_contours
```

```python
viewer = napari.Viewer()

def screenshot() -> None:
   display(nbscreenshot(viewer))
```

## Load Data

Load the Fluo-C2DL-Huh7 dataset.


```python
dataset = "01"
data_path = Path("Fluo-C2DL-Huh7") / dataset
image = imread(str(data_path / "*.tif"))

viewer.add_image(image)
screenshot()
```

## Configuration

We'll use the same configuration as in the previous example, except for `config.segmentation_config.min_frontier` which had its value decreased.

The `min_frontier` merges regions with an average contour lower than the provided value.
Since the contours are combined by averaging, the previous value of 0.1 removed relevant segments from the candidate hypotheses.

As a reminder, the configuration parameters documentation can be found [here](https://github.com/royerlab/ultrack/blob/main/ultrack/config/README.md).


```python
config = MainConfig()

# Candidate segmentation parameters
config.segmentation_config.n_workers = 7
config.segmentation_config.min_area = 2500
config.segmentation_config.min_frontier = 0.05  # NOTE: this parameter is not the same as in intro.ipynb

# Setting the maximum number of candidate neighbors and maximum spatial distance between cells
config.linking_config.max_neighbors = 5
config.linking_config.max_distance = 100
config.linking_config.n_workers = 7

# Adding absurd weight to division because there's no diving cell
config.tracking_config.division_weight = -100
# Very few tracks enter/leave the field of view, increasing penalization
config.tracking_config.disappear_weight = -1
config.tracking_config.appear_weight = -1

print(config)
```

## Cellpose Segmentation

The same function as `intro.ipynb` to segment cells within each frame.


```python
cellpose = Cellpose(model_type="cyto2", gpu=True)

def predict(frame: ArrayLike, gamma: float) -> ArrayLike:
    norm_frame = normalize(np.asarray(frame), gamma=gamma)
    return cellpose(norm_frame, tile=False, normalize=False, diameter=75.0)
```

## Metrics

Helper function to evaluate tracking score using [Cell Tracking Challenge](celltrackingchallenge.net)'s metrics and annotations.


```python
def score(output_path: Path) -> Dict:
    gt_path = data_path.parent / f"{data_path.name}_GT"
    return evaluate_sequence(
        str(output_path.absolute()),
        str(gt_path.absolute()),
        metrics=["TRA", "CHOTA"],
    )
```

## Parameter Search

Here, we evaluate the segmentation and tracking given multiple values of `gamma`, used on the normalization step before the Cellpose prediction.


```python
all_labels = []
metrics = []
gammas = [0.1, 0.25, 0.5, 1]
sigma = 5.0

tracker = Tracker(config)

for gamma in gammas:

    # cellpose prediction
    cellpose_labels = np.zeros_like(image, dtype=np.int32)
    array_apply(
        image,
        out_array=cellpose_labels,
        func=predict,
        gamma=gamma,
    )
    all_labels.append(cellpose_labels)
    
    name = f"{dataset}_labels_{str(gamma).replace(".", '_')}"
    viewer.add_labels(cellpose_labels, name=name, visible=False)

    # cell tracking using `labels` parameter, it's the same as using `labels_to_edges`.
    tracker.track(
        labels=cellpose_labels,
        sigma=sigma,
        overwrite=True
    )

    # exporting to CTC format
    output_path = data_path.parent / Path(name.upper()) / "TRA"
    tracker.to_ctc(output_path, overwrite=True)

    # computing tracking score
    metric = score(output_path)
    metric["gamma"] = gamma
    metrics.append(metric)

print(metrics)
```

## Combined Contours and Detection

The `labels_to_contours` combines multiple segmentation labels into a single detection and contour map.

The detection map is the maximum value between the binary masks of each label.

The contour map is the average contour map of the binary contours of each label.


```python
foreground, contours = labels_to_contours(all_labels, sigma=sigma)
```


```python
foreground
```

```python
layer = viewer.add_labels(foreground.astype(int))
screenshot()
layer.visible = False
```


```python
layer = viewer.add_image(contours)
screenshot()
layer.visible = False
```

## Tracking

Run the tracking algorithm on the provided configuration, and combined detections and contours.


```python
tracker.track(
   foreground=foreground,
   contours=contours,
   overwrite=True
)
```

Compute metrics for the multiple hypotheses tracking and compare the scores of the different approaches.


```python
output_path = Path(f"{dataset}_COMBINED") / "TRA"
tracker.to_ctc(output_path, overwrite=True)

metric = score(output_path)
metrics.append(metric)

df = pd.DataFrame(metrics)
df.to_csv(f"{dataset}_scores.csv", index=False)
df
```

## Exporting and Visualization

The intermediate tracking data is stored on disk and must be exported to your preferred format.
Here we convert the resulting tracks to a DataFrame and Zarr to visualize using napari if running locally.


```python
tracks_df, graph = tracker.to_tracks_layer()
tracks_df.to_csv(f"{dataset}_tracks.csv", index=False)

segments = tracker.to_zarr(
    overwrite=True,
)

viewer.add_tracks(
    tracks_df[["track_id", "t", "y", "x"]],
    name="tracks",
    graph=graph,
    visible=True,
)

viewer.add_labels(segments, name="segments").contour = 2
screenshot()
```
