import yaml


class Configs:
    """
    Defines hyperparameters for the model training process.
    """
    def __init__(self, config_path):
        config = self.load_yaml(config_path)

        for key, value in config.items():
            setattr(self, key, value)

    @staticmethod
    def load_yaml(yaml_path):
        """
        Loads hyperparameters from a YAML configuration file.
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    