import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

torch.manual_seed(0)


bay_train_writer_layout = \
    lambda task_id: {
        f"Task{task_id}": {
            "loss": ["Multiline", [f"task{task_id}/loss/total", f"task{task_id}/loss/NLL", f"task{task_id}/loss/KL"]],
            "accuracy": ["Multiline", [f"task{task_id}/acc/test", f"task{task_id}/acc/train"]],
            "KL": ["Multiline", [f"task{task_id}/loss/KL/const_term", f"task{task_id}/loss/KL/log_std_term", f"task{task_id}/loss/KL/mu_diff_term", f"task{task_id}/loss/KL/std_quotient_term"]],
        }
    }


def display_img(x: torch.Tensor):
    """
    Display the given image, assume x is a (HxW) image and is in the range [-1,1]
    """
    plt.imshow(x.detach().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()


SETS_0 = [0, 2, 4, 6, 8]
SETS_1 = [1, 3, 5, 7, 9]
def id_to_idxs(task_id):
    return (SETS_0[task_id], SETS_1[task_id])