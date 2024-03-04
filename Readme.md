# IDML-2 UNet for defective steel - image segmentation

## Introductions
 The primary objective of this project was to construct a segmentation model capable of segmenting defective regions from the images of steel sheets. The dataset provided for this task con sist of around 12000 images of 256*1600 size, with annotations file. Several preprocessing steps were employed to prepare the dataset for modeling.

## Summary:

The goal of this project was to train a U-Net model for steel defect image segmentation. Two variations of the U-Net architecture were implemented:

1. **Baseline U-Net**: This model used a simple CNN as the encoder.
2. **Complex U-Net**: This model utilized ResNet18 with pre-trained weights as the encoder.

### Evaluation Metrics:
Both models were evaluated using Intersection over Union (IoU) and Dice score metrics.

### Performance Comparison:
The complex U-Net model with ResNet18 as the encoder outperformed the baseline U-Net model. It achieved a Dice score of around 0.7, indicating superior segmentation performance.

### Testing Phase:
The complex U-Net model was tested on unseen images. It produced satisfactory segmentation masks, demonstrating its ability to generalize well to new data.

Overall, the project successfully demonstrates the effectiveness of using a more complex architecture (U-Net with ResNet18) for steel defect image segmentation tasks. The model's performance was assessed using standard evaluation metrics and validated through testing on unseen images, showcasing its potential for practical applications in steel defect detection and segmentation.

## Directories

The project is structured as follows:

- **bin**: This directory contains the source code files required to run the analysis.
  - `1. unet-eda-and-modelling-baseline`: This Python notebook is the main entry point for executing the project. It contains the code that reads the dataset and performs various data processing, modelling and visualization tasks.
  - `1. 2. unet-tesing-and-predictions`: This Python notebook demonstrates the testing and prediction phase for the constructed Unet model.
 

- **scripts**: This directory contains the script of models used for this projects (Baseline and superior Unets).

- **plots**: This directory contains the plots produced during this project.
## User Guide
To use this project, follow the steps below:

1. Clone the repository: git clone https://github.com/fahaddeshmukh/steel-defect


2. Install the required dependencies: pip install -r requirements.txt



3. Prepare the dataset:
- Download the dataset from [here](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) (Kaggle Steel Defect Segmentation Competition).


4. Run the mentioned notebooks



## Contact Information
For any questions, suggestions, or issues, please feel free to contact:

- Name: Fahad Deshmukh

- Email: deshmukh@uni-potsdam.de
