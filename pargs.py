import torch


class Arguments:
    # Class used to store parameters

    def __init__(self, name):
        self.name = name
        self.max_seq_length = 256
        self.output_mode = "classification"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.seed = 1
        self.train_batch_size = 32
        self.eval_batch_size = 1
        self.num_train_epochs = 1
        self.adam_epsilon = 1e-8
        self.model_name_or_path = "bert-base-uncased"
        self.max_grad_norm = 1.0
        self.logging_steps = 400
        self.save_steps = 200000
        self.eval_all_checkpoints = False
        self.do_lower_case = False
        self.is_incremental = False
        self.gradient_accumulation_steps = 1
        self.task_name = 'document-level-sa'
        self.label_list = []
        self.amply = 10