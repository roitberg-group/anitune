class FinetuneSpec:
    def __init__(
        self,
        num_head_layers: int = 1,
        head_lr: float = 1e-3,
        body_lr: float = 0.0,
        unfreeze_body_epoch_num: int = 0,
    ) -> None:
        self.num_head_layers = num_head_layers
        self.head_lr = head_lr
        self.body_lr = body_lr
        self.unfreeze_body_epoch_num = unfreeze_body_epoch_num
        if self.head_lr <= 0.0:
            raise ValueError(
                "Learning rate for the head of the model must be strictly positive"
            )
        if self.body_lr < 0.0:
            raise ValueError(
                "Learning rate for the body of the model must be positive or zero"
            )

    @property
    def frozen_body(self) -> bool:
        return self.body_lr == 0.0
