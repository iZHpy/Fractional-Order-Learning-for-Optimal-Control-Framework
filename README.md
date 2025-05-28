# Fractional-Order Learning for Optimal Control Framework

This project implements a control framework based on fractional-order learning, using deep learning models to predict and optimize control strategies.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Fractional-Order-Learning-for-Control-Framework.git
    cd Fractional-Order-Learning-for-Control-Framework
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    conda env create -f environment.yml
    conda activate FOLOC
    ```

3. Install dependencies:
    ```bash
    pip install 
    ```

## Usage

1. Configuration

    In the `configs` folder, you can find a sample configuration file `configs.yaml`. Modify the configuration file according to your needs.

2. Train the model

    Run the following command to start training the model:
    ```bash
    python main.py --config ./configs/configs.yaml
    ```

3. Evaluate the model

    After training, the model will automatically evaluate on the test set and output the evaluation results.

## File Structure

- `main.py`: Main program entry, responsible for loading configurations, initializing the model, training, and evaluation.
- `loader/Dataloader.py`: Data loading and preprocessing module.
- `layer/models.py`: Defines various deep learning models.
- `layer/layers.py`: Defines the layers used in the models.
- `configs`: Stores configuration files.
- `utils`: Stores utility functions.

## Contributing

Issues and pull requests are welcome. If you have any suggestions or improvements, please feel free to contact us.
