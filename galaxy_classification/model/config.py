import yaml
from galaxy_classification.model.galaxy_cnn import GalaxyClassificationCNNConfig, GalaxyRegressionCNNConfig

def load_config_from_yaml(path: str) -> GalaxyClassificationCNNConfig:
    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    return GalaxyClassificationCNNConfig(**config_dict)

def load_regression_config_from_yaml(path: str) -> GalaxyRegressionCNNConfig:
    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)
    return GalaxyRegressionCNNConfig(**config_dict)