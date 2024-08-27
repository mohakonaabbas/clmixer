
<div align="center">

# CLMixer: A Practical Approach to Combine Building Blocks of a Continual Learning Algorithm

</div>

**CLMixer** is an innovative continual learning library inspired by the philosophy of [Avalanche](https://avalanche.continualai.org), extending its capabilities to support new taxonomy and industrial applications. CLMixer is designed to make it easy to dynamically create, test, and optimize continual learning methods, even for non-programmers.

---

## Key Features

- **JSON-Based Parametrization**: Define complex continual learning workflows using simple JSON configuration files.
- **Dynamic Method Creation**: Automatically generate custom continual learning methods based on JSON configurations.
- **Automatic Architecture Discovery**: Seamlessly search for and discover the best architectures suited for your datasets.

---

## JSON-Based Parametrization and Dynamic Method Creation

At the core of CLMixer is a JSON-based system that allows users to define operations, hyperparameters, and even entire methods without writing code. Here's a glimpse of what the JSON configuration looks like:

### Optimization Configuration Example

```json
{
    "optimisation": {
        "lr": 0.001,
        "epochs": 1000,
        "optimizer": {
            "type": "adam",
            "weight_decay": 0.0005
        },
        "batch_size": 32,
        "device": "cuda"
    }
}
```

### Model and Data Configuration Example

```json
{
    "model": {
        "model_type": "dinov2_vits14",
        "hidden_size": 20
    },
    "data": {
        "dataset_name": "kth",
        "data_path": "path/to/kth",
        "n_experiments": 5,
        "backbone": "None",
        "scenario": "indus_cil"
    }
}
```

### Plugins Configuration Example

```json
{
    "plugins": [
        {
            "name": "RandomMemoryUpdaterOperation",
            "hyperparameters": {
                "cls_budget": 10
            },
            "function": "knowledge_retention"
        },
        {
            "name": "CrossEntropyOperation",
            "hyperparameters": {},
            "function": "knowledge_incorporation"
        },
        {
            "name": "KnowledgeDistillationOperation",
            "hyperparameters": {
                "temperature": 2.0
            },
            "function": "knowledge_retention"
        },
        {
            "name": "FinetuneOperation",
            "hyperparameters": {
                "finetune_epochs": 100,
                "finetune_bs": 32,
                "finetune_lr": 0.001,
                "cls_budget": 15
            },
            "function": "bias_mitigation"
        }
    ]
}
```

### Internal Plugin Example: KnowledgeDistillationOperation

```python
class KnowledgeDistillationOperation(Operation):
    def __init__(self,
                entry_point=["before_backward","after_eval_forward"],
                inputs={},
                callback=(lambda x:x), 
                paper_ref="",
                is_loss=True):
        super().__init__(entry_point, inputs, callback, paper_ref, is_loss)

        self.set_callback(self.kd_callback)
        self.set_config_template({
            "name": self.__class__.__name__,
            "hyperparameters": {
                "temperature": 2.0
            },
            "function": "knowledge_retention"
        })
```

---

## Automatic Architecture Discovery

CLMixer's automatic architecture discovery feature enables users to explore various combinations of plugins, models, and optimizers to find the best solution for their dataset. This feature allows for:

- **Rapid Prototyping**: Quickly test different configurations by adjusting JSON parameters.
- **Automatic Optimization**: Discover optimal architectures tailored to the specific characteristics of your data.

---

## Advantages of CLMixer

1. **Ease of Use**: By using JSON configurations, users can design and modify continual learning workflows without deep programming knowledge.
2. **Flexibility**: The dynamic method creation allows for experimenting with various continual learning strategies.
3. **Scalability**: The architecture discovery feature facilitates scalable and automated optimization, making it suitable for a wide range of applications.

---

CLMixer brings a practical, user-friendly approach to continual learning, making it accessible to both researchers and industry practitioners. Whether you're looking to build custom learning algorithms or optimize existing ones, CLMixer provides the tools you need to succeed.