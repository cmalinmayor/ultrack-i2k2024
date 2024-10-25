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

# Multi-color cell tracking

+++

Now you are on your own! Try to apply the concepts you learned in the previous sections to a new dataset.

After we go through the dataset, we will provide some cues on how we would approach the problem.

Feel free to ask for help if you get stuck or want to discuss your results.

## Download data

We will download the data from our public server. The dataset is metastic breat cancer cells with RGB lentiviral marking, data from [Lammerding lab](https://lammerding.wicmb.cornell.edu/).

```{code-cell} ipython3
:tags: [remove-output]

!wget -nc https://public.czbiohub.org/royerlab/ultrack/multi-color-cytoplasm.tif
```

## Import libraries

We will start with the minimal import of `napari` and `tifffile`, update this chunk as needed.

```{code-cell} ipython3
import napari
from napari.utils import nbscreenshot
from tifffile import imread
```

## Setting up napari viewer

Napari is setup the same way as the previous tutorial.

```{code-cell} ipython3
viewer = napari.Viewer()
viewer.window.resize(1800, 1200)

def screenshot() -> None:
   display(nbscreenshot(viewer))
```

## Data loading

We will load the data and display in the viewer as an RGB image.

```{code-cell} ipython3
image = imread("multi-color-cytoplasm.tif")
print("Image array shape", image.shape)

viewer.add_image(image, rgb=True, name="multi-color-cytoplasm")
screenshot()
```

```{tip}
The image stack have 3 channels, since we don't have multi-color segmentation model (or at least I don't), we could process each channel individually and then combine them using Ultrack's to obtain a final segmentation and tracking without complicated heuristics for multi-color tracking.
```

## Your turn!
