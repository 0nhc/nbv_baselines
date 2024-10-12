import argparse
import os.path
import shutil
import yaml


class ConfigManager:
    config = None
    config_path = None

    @staticmethod
    def get(*args):
        result = ConfigManager.config
        for arg in args:
            result = result[arg]
        return result

    @staticmethod
    def load_config_with(config_file_path):
        ConfigManager.config_path = config_file_path
        if not os.path.exists(ConfigManager.config_path):
            raise ValueError(f"Config file <{config_file_path}> does not exist")
        with open(config_file_path, 'r') as file:
            ConfigManager.config = yaml.safe_load(file)

    @staticmethod
    def backup_config_to(target_config_dir, file_name, prefix="config"):
        file_name = f"{prefix}_{file_name}.yaml"
        target_config_file_path = str(os.path.join(target_config_dir, file_name))
        shutil.copy(ConfigManager.config_path, target_config_file_path)

    @staticmethod
    def load_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='', help='config file path')
        args = parser.parse_args()
        if args.config:
            ConfigManager.load_config_with(args.config)

    @staticmethod
    def print_config(key: str = None, group: dict = None, level=0):
        table_size = 80
        if key and group:
            value = group[key]
            if type(value) is dict:
                print("\t" * level + f"+-{key}:")
                for k in value:
                    ConfigManager.print_config(k, value, level=level + 1)
            else:
                print("\t" * level + f"| {key}: {value}")
        elif key:
            ConfigManager.print_config(key, ConfigManager.config, level=level)
        else:
            print("+" + "-" * table_size + "+")
            print(f"| Configurations in <{ConfigManager.config_path}>:")
            print("+" + "-" * table_size + "+")
            for key in ConfigManager.config:
                ConfigManager.print_config(key, level=level + 1)
            print("+" + "-" * table_size + "+")


''' ------------ Debug ------------ '''
if __name__ == "__main__":
    test_args = ['--config', 'local_train_config.yaml']
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--config', type=str, default='', help='config file path')
    test_args = test_parser.parse_args(test_args)
    if test_args.config:
        ConfigManager.load_config_with(test_args.config)
    ConfigManager.print_config()
    print()
    pipeline = ConfigManager.get('settings', 'train', 'batch_size')
    ConfigManager.print_config('settings')
    print(pipeline)
