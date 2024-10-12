import torch


class TensorboardWriter:
    @staticmethod
    def write_tensorboard(writer, panel, data_dict, step):
        complex_dict = False
        if "scalars" in data_dict:
            scalar_data_dict = data_dict["scalars"]
            TensorboardWriter.write_scalar_tensorboard(writer, panel, scalar_data_dict, step)
            complex_dict = True
        if "images" in data_dict:
            image_data_dict = data_dict["images"]
            TensorboardWriter.write_image_tensorboard(writer, panel, image_data_dict, step)
            complex_dict = True
        if "points" in data_dict:
            point_data_dict = data_dict["points"]
            TensorboardWriter.write_points_tensorboard(writer, panel, point_data_dict, step)
            complex_dict = True

        if not complex_dict:
            TensorboardWriter.write_scalar_tensorboard(writer, panel, data_dict, step)

    @staticmethod
    def write_scalar_tensorboard(writer, panel, data_dict, step):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                writer.add_scalars(f'{panel}/{key}', value, step)
            else:
                writer.add_scalar(f'{panel}/{key}', value, step)

    @staticmethod
    def write_image_tensorboard(writer, panel, data_dict, step):
        pass

    @staticmethod
    def write_points_tensorboard(writer, panel, data_dict, step):
        for key, value in data_dict.items():
            if value.shape[-1] == 3:
                colors = torch.zeros_like(value)
                vertices = torch.cat([value, colors], dim=-1)
            elif value.shape[-1] == 6:
                vertices = value
            else:
                raise ValueError(f'Unexpected value shape: {value.shape}')
            faces = None
            writer.add_mesh(f'{panel}/{key}', vertices=vertices, faces=faces, global_step=step)
