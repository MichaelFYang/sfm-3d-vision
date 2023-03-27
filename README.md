# Structure from Motion (SfM) Project - 3D Vision Project

This is a Python project that implements a simple SfM pipeline. (TO be update... using pytorch ...) The pipeline consists of the following steps:

1. Camera calibration using a set of calibration images
2. Feature extraction and matching between pairs of images
3. Triangulation of 3D points from the matched features

## Installation

To run this project, you will need Python 3 and the following libraries:

- NumPy (should be replaced)
- OpenCV (should be replaced)
- Matplotlib
- TODO ...

You can install these libraries using pip:

    pip install numpy opencv-python matplotlib

## Usage

To run the SfM pipeline, you can use the `main.py` script:


This will perform camera calibration using the calibration images in the `calibration_images/` directory, and then reconstruct 3D points from the images in the `images/` directory.

The reconstructed 3D points will be visualized using a 3D scatter plot.

TODO ...

## Customization

You can customize the SfM pipeline by modifying the following:

- Calibration pattern size: you can change the pattern size by modifying the `pattern_size` variable in `calibrator.py`
- Feature extraction method: you can change the feature extraction method by modifying the `method` parameter in `FeatureExtractor` class in `feature_extractor.py`
- Feature matching method: you can change the feature matching method by modifying the `matcher` parameter in `FeatureMatcher` class in `feature_extractor.py`

## Credits

This project was inspired by the following resources:

- [OpenCV SfM module](https://docs.opencv.org/4.x/d9/d0c/group__sfm.html)
- [Pytorch](https://pytorch.org/)
- [TODO](https:link)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.