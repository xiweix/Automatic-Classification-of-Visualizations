# A Machine Learning Approach to the Automatic Classification of Visualizations

Code for the project "A Machine Learning Approach to the Automatic Classification of Visualizations". ([Project report](https://drive.google.com/file/d/1sh2TTRITh4ECNBizn6PwkPjJonxsNfO9/view "Project report"))

### Dataset - Beagle
Beagle includes over 41,000 visualizations across five different tools and repositories extracted from web.
[Dataset resource] [Beagle: Automated Extraction and Interpretation of Visualizations from the Web.](https://homes.cs.washington.edu/~leibatt/beagle.html "Beagle: Automated Extraction and Interpretation of Visualizations from the Web.")

### GradCAM
We use Grad-CAM implementation for PyTorch from this [repository](https://github.com/jacobgil/pytorch-grad-cam "repository").

### Environment Setup and Run
* (recommended) Anaconda https://docs.anaconda.com/anaconda/install/index.html
* (recommended) GPU - CUDA
* Python >= 3.7
* PyTorch https://pytorch.org/get-started/locally/
* Download the dataset and save it under folder "/dataset"
* Check the dataset and save information into a csv file:
`python get_csv.py`
* Train (and save) the model
`python3 train.py --model-name densenet121 --batch-size 128 --val-batch-size 200 | tee densenet121_128.txt`
* Grad-CAM explanations can be generated by:
`python gradcam_main.py`