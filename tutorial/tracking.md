---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 2D Cell Tracking with Multiple Hypotheses

+++

This tutorial demonstrates Ultrack's multiple hypotheses tracking capabilities in a 2D cell tracking example due to time constraints. The same principles apply to 3D tracking.

We will start with a simple classical image processing solution using Ultrack to track cells based on image intensities directly. Then, we will progress to using Cellpose and finally use multiple Cellpose segmentations to improve tracking performance without training our own segmentation model, showcasing the improvement when a single segmentation isn't good enough.

## Download Data

Download the Fluo-C2DL-Huh7 dataset from the [Cell Tracking Challenge](https://celltrackingchallenge.net), which contains fluorescence microscopy images for cell tracking.

The dataset will be used to demonstrate the segmentation and tracking workflow.

```{code-cell} ipython3
:tags: [remove-output]

!wget -nc http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip
!unzip -n Fluo-C2DL-Huh7.zip
```

## Import libraries

Import the libraries needed for reading images, processing them, cell segmentation, tracking, and performance metrics.

```{code-cell} ipython3
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import napari
import scipy.ndimage as ndi

from ctc_metrics import evaluate_sequence
from dask.array.image import imread
from napari.utils import nbscreenshot
from numpy.typing import ArrayLike
from rich import print
from skimage import filters
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from edt import edt

from ultrack import Tracker, MainConfig
from ultrack.imgproc import normalize, Cellpose
from ultrack.utils.array import array_apply
from ultrack.utils import labels_to_contours
```

## Setting up napari viewer

We use napari to visualize the data and the results, Ultrack can also export to other formats such as TrackMate, so you can use your preferred visualization tool in your workflow.

We resize the viewer so the screenshots are consistent between runs.

```{code-cell} ipython3
viewer = napari.Viewer()
viewer.window.resize(1800, 1200)

def screenshot() -> None:
   display(nbscreenshot(viewer))
```

## Data loading

Load the Fluo-C2DL-Huh7 frames as a (T, Y, X) array.

```{code-cell} ipython3
dataset = "02"
data_path = Path("Fluo-C2DL-Huh7") / dataset
image = np.asarray(imread(str(data_path / "*.tif")))

viewer.add_image(image, colormap="magma")
viewer.dims.set_point(0, 28)
screenshot()
```

## Example of classical segmentation without Ultrack

Classical tracking usually involves segmenting the cells and then tracking them using the segmentation as input.

Here we start with a simple segmentation using a Gaussian filter to remove noise, a manually set threshold to create a binary mask, and a watershed transform to separate the cells.

```{code-cell} ipython3
blurred_image = filters.gaussian(image, (0, 5, 5))
layer = viewer.add_image(blurred_image, colormap="green", blending="additive")
screenshot()
layer.visible = False
```

We apply a manual thresholding to create a binary mask. The value of 0.1 was chosen by visual inspection.

```{code-cell} ipython3
threshold_foreground = blurred_image > 0.1
layer = viewer.add_labels(threshold_foreground)
screenshot()
layer.visible = False
```

We use the watershed transform to separate the cells, using the distance transform because the cells are somewhat convex, the tricky part is setting the `h` parameter, which controls the minimum depth of the minima. The more we increase `h`, the more we merge regions, resulting in fewer segments.

```{admonition} Interaction
Play with the `h` parameter to see how it affects the segmentation.
```

```{code-cell} ipython3
h = 10

def ws_from_h_minima(mask: np.ndarray) -> np.ndarray:
    dist = edt(mask)
    maximas = h_maxima(dist, h=h)
    markers, _ = ndi.label(maximas)
    return watershed(-dist, markers=markers, mask=mask)

ws_segm = np.zeros_like(threshold_foreground, dtype=np.int16)

array_apply(
    threshold_foreground,
    func=ws_from_h_minima,
    # blurred_image,
    # func=lambda mask, img: watershed(-img, mask=mask),
    out_array=ws_segm,
)
layer = viewer.add_labels(ws_segm, name=f"watershed h={h}")
screenshot()
layer.visible = False
```

This is not only exclusive to watershed, but also to other segmentation algorithms, have parameters which might be hard to set and vary within your dataset, for example if they are related to cell size, image quality, or other factors.

+++

## Ultracking & segmentation from image intensities directly

In this section, we will show how the temporal information of your data can be used to bypass some of the segmentation parameters, letting you choose the segmentations that are most consistent over time.

Ultrack main interface is the `Tracker` class, which exposes a `track` method that computes the tracks given the input data.

The `Tracker` class is configured with a `MainConfig` object, which contains the parameters for the segmentation, linking, and tracking steps, which will setup next and its documentation can be found [here](https://royerlab.github.io/ultrack/configuration.html).

### Configuration

```{code-cell} ipython3
print(MainConfig())
```

The minimum area is an important parameter, otherwise we will have too many hypotheses to track, we can easily find this information from the previous segmentation.

```{admonition} Interaction
Select a cell and a time point in napari, edit the cell_id and cell_time_point variables below, and run the cell to find the size of the cell.
```

```{code-cell} ipython3
:tags: [no-execute]

cell_id = 61
cell_time_point = 21

size = (viewer.layers["watershed h=10"].data[cell_time_point] == cell_id).sum()
size
```

Next we create the configuration object and set the segmentation parameters, we will come back to the tracking parameters later, most parameter are interpretable together with the imaging data, with the exception of the tracking weights.

```{admonition} Interaction
Edit the `min_area` parameter with the `size` variable computed above.

Update the `max_distance` parameter with a guessestimate from napari, error on the side of overestimating.
```

```{code-cell} ipython3
config = MainConfig()

# Candidate segmentation parameters
config.segmentation_config.n_workers = 7
config.segmentation_config.min_area = 2500  # UPDATE: use `size` variable

# Other important parameters depending on your data
# config.segmentation_config.max_area = ...

# maximum movement between frames
config.linking_config.max_distance = 100  # UPDATE: update inspecting the moving cells.

# UPDATE: we will come back here to select these parameters
# Tracking penalization parameters
config.tracking_config.disappear_weight = -0.5
config.tracking_config.appear_weight = -0.5
config.tracking_config.division_weight = -1

print(config)
```

With the configuration we create our tracker object, and using the previously computed foreground and blurred images which provide cues for the cell boundaries, we can segment and track the cells.

```{code-cell} ipython3
:tags: [remove-output]

tracker = Tracker(config)

tracker.track(foreground=threshold_foreground, contours=-blurred_image, overwrite=True)

classic_segm = tracker.to_zarr()

layer = viewer.add_labels(classic_segm)
screenshot()
layer.visible = False
```

```{admonition} Interaction
Try changing ultrack's parameters and see how it affects the segmentation and tracking.
Feel free to ask for help understanding how they affect the results.
```

```{warning}
Don't forget the `overwrite=True` when tracking.
Ultrack was designed to track datasets larger than memory so by default it stores intermediate results on disk.
```

While this approach provides a good starting point and highlights how classical pipelines can be improved with Ultrack, it isn't making use of existing pre-trained models, which can provide better segmentations and are often available for 2D data. So in the next section, we will use [Cellpose](https://github.com/MouseLand/cellpose) to segment the cells and track them.

## Cellpose segmentation

First, we define a function to predict the segmentation using Cellpose and the `cyto2` model.
The `diameter` parameter is set to 75.0, which is a good starting point for this dataset.
We also disable the normalization step in Cellpose, as we are doing our own normalization using the `gamma` parameter for contrast adjustment.

```{code-cell} ipython3
:tags: [remove-output]

cellpose = Cellpose(model_type="cyto2", gpu=True)

def predict(frame: ArrayLike, gamma: float) -> ArrayLike:
    norm_frame = normalize(np.asarray(frame), gamma=gamma)
    return cellpose(norm_frame, tile=False, normalize=False, diameter=75.0)
```

With the `predict` function defined, we apply it to all frames.

```{code-cell} ipython3
gamma = 1.0

cellpose_labels = np.zeros_like(image, dtype=np.int32)
array_apply(
    image,
    out_array=cellpose_labels,
    func=predict,
    gamma=gamma,
)

viewer.add_labels(cellpose_labels, name=f"cellpose gamma={gamma}", visible=False)
```

Next, we use the `labels` parameter to track the cells from the Cellpose segmentation.

Because we are using labels we want to merge regions with small frontiers (no contours within segments) removing them from the hierarchy.

```{code-cell} ipython3
:tags: [remove-output]

sigma = 5.0  # optional sigma parameter for smoothing labels' contours

config.segmentation_config.min_frontier = 0.1   # adding min. frontier to merge regions within segments

tracker = Tracker(config)

tracker.track(labels=cellpose_labels, sigma=sigma, overwrite=True)

tracks_df, graph = tracker.to_tracks_layer()
segm = tracker.to_zarr()

tracks_layer = viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]])
lb_layer = viewer.add_labels(segm, name=f"labels gamma={gamma}")

screenshot()

tracks_layer.visible = False
lb_layer.visible=False
```

This approach provides a better segmentation than the classical approach, but it still has limitations, as the segmentation is not perfect, especially in dimmer cells or cells with low contrast.

Next, we will evaluate if tuning the `gamma` parameter can improve Cellpose's segmentation and therefore our tracking results.

## Metrics

Because this dataset is part of the [Cell Tracking Challenge (CTC)](celltrackingchallenge.net), we can evaluate the tracking performance using the provided ground truth annotations.
For this, we define a helper function to evaluate tracking score CTC's tracking metric (TRA) with the [ctc-metrics](https://github.com/CellTrackingChallenge/py-ctcmetrics) package.

```{code-cell} ipython3
def score(output_path: Path) -> Dict:
    gt_path = data_path.parent / f"{data_path.name}_GT"
    return evaluate_sequence(
        str(output_path.absolute()),
        str(gt_path.absolute()),
        metrics=["TRA"],
    )
```

## Parameter (gamma) Search

We will evaluate the segmentation and tracking performance using different values of `gamma` to adjust the contrast of the input images to Cellpose.
Smaller `gamma` decreases the difference in brighter regions, highlighting dimmer cells at the cost of saturating the brighter cells.

```{code-cell} ipython3
:tags: [remove-output]

gammas = [0.1, 0.25, 0.5, 1]
all_labels = []
metrics = []

# iterating over different gamma values
for gamma in gammas:

    # Cellpose prediction
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

    # exporting to CTC format for evaluation
    output_path = data_path.parent / Path(name.upper()) / "TRA"
    tracker.to_ctc(output_path, overwrite=True)

    # computing tracking score and storing it
    metric = score(output_path)
    metric["gamma"] = gamma
    metrics.append(metric)
```

```{code-cell} ipython3
:tags: [remove-input]

# printing the results
pd.DataFrame(metrics).sort_values("TRA", ascending=False)
```

Here, we can see `gamma=0.5` provides the best tracking performance.
However, when inspecting visually we can see that some of the mistakes are complementary between the different segmentations, so we can combine them to obtain an improve the tracking performance.

## Combined contours and foreground

Because Ultrack operates on the contours of the segmentation, combine the segmentations labels through their contour and foreground maps.

To do that, we provide the `labels_to_contours`, where the contours are the average of the binary contours of each label, and the foreground is the maximum value between the binary masks of each label.

```{code-cell} ipython3
foreground, contours = labels_to_contours(all_labels, sigma=sigma)
```

```{code-cell} ipython3
layer = viewer.add_labels(foreground.astype(int))
screenshot()
layer.visible = False
```

```{code-cell} ipython3
layer = viewer.add_image(contours)
screenshot()
layer.visible = False
```

## Tracking

Once we have the combined contours and foreground, the process is the same as every before.

```{code-cell} ipython3
:tags: [remove-output]

tracker.track(
   foreground=foreground,
   contours=contours,
   overwrite=True
)
```

Once again, we compute metrics for the multiple hypotheses tracking and compare the scores of the different approaches.

```{code-cell} ipython3
output_path = Path(f"{dataset}_COMBINED") / "TRA"
tracker.to_ctc(output_path, overwrite=True)

metric = score(output_path)
metric["gamma"] = "all"
metrics.append(metric)

df = pd.DataFrame(metrics)
df.to_csv(f"{dataset}_scores.csv", index=False)

df.sort_values("TRA", ascending=False)
```

The combined segmentation provides an improved tracking performance, even when compared to the best individual segmentation, showcasing the benefits of using multiple segmentations to improve tracking results.

This could be further adapted to other parameters or combining different segmentation algorithms to obtain a larger diversity of segmentations.

## Exporting and visualization

To further assess our results or downstream analysis, we can export the tracks and segmentations to a CSV file and a zarr file, respectively, for further analysis and visualization.

```{code-cell} ipython3
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

viewer.add_labels(segments, name="segments", opacity=0.5)

screenshot()
```
