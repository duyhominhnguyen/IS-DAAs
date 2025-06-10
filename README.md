# Mitigating Reward Over-optimization in Direct Alignment Algorithms with Importance Sampling

## Abstract

Direct Alignment Algorithms (DAAs) such as Direct Preference Optimization (DPO) have emerged as alternatives to the standard Reinforcement Learning from Human Feedback (RLHF) for aligning large language models (LLMs) with human values. However, these methods are more susceptible to over-optimization, in which the model drifts away from the reference policy, leading to degraded performance as training progresses. This paper proposes a novel importance-sampling approach to mitigate the over-optimization problem of offline DAAs. This approach, called IS-DAAs, multiplies the DAA objective with an importance ratio that accounts for the reference policy distribution. IS-DAAs additionally avoid the high variance issue associated with importance sampling by clipping the importance ratio to a maximum value. Our extensive experiments demonstrate that IS-DAAs can effectively mitigate over-optimization, especially under low regularization strength, and achieve better performance than other methods designed to address this problem.

![Overview](figures/fig_1.png)

## Repository Structure

```

├── README.md
├── requirements.txt
├── src
│   ├── data
│   │   ├── summarization
│   │   └── instruction_following
│   ├── models
│   │   ├── llama_3.2_3B
│   │   └── fine_tuned_models
│   ├── scripts
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── utils.py
│   └── baselines
│       ├── rpo
│       ├── chi_po
│       └── length_regularization
└── experiments
    ├── summarization
    └── instruction_following
```

## Requirements

To run the code in this repository, you need to install the following dependencies:

```bash
pip install -r requirements.txt
```

## Experiments

In this section, we empirically evaluate IS-DAAs' ability to align language models with human preferences and mitigate the reward over-optimization problem.

### TL;DR Summarization

We systematically study the trade-off between the policy performance and KL regularization achieved by different alignment methods in a controlled environment where we assume to have access to a golden reward model as the ground-truth preferences.

### Instruction Following

We evaluate IS-DAAs on three standard open-ended instruction following benchmarks. Under both settings, IS-DAAs outperform existing alignment approaches and better mitigate the over-optimization problem compared to existing approaches designed for this purpose.

### Models

Throughout our experiments, we use Llama-3.2-3B as the pre-trained base model. For both summarization and instruction following, we first supervised fine-tuning Llama-3.2-3B to serve as the initialization for subsequent preference training.

### Baselines

In addition to IS-DAAs, we evaluate several existing baselines that address the over-optimization problem in DAAs, including:

- **DAAs+SFT objective**: Augments the DAAs with an additional SFT loss, referred to as Regularized Preference Optimization (RPO).
- **χ-PO**: Combines χ² with KL regularization to enforce stronger regularization.
- **DAAs with length-regularization**: Addresses the length exploitation issue, a common pattern of reward over-optimization.

## Usage

### Training

To train the model, run the following command:

```bash
python src/scripts/train.py --config config.yaml
```

### Evaluation

To evaluate the model, run the following command:

```bash
python src/scripts/evaluate.py --model_path path/to/model
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper,
  title={Mitigating Reward Over-optimization in Direct
Alignment Algorithms with Importance Sampling},
  author={Phuc Minh Nguyen, Ngoc-Hieu Nguyen, Duy H. M. Nguyen, Anji Liu, An Mai, Binh T. Nguyen, Daniel Sonntag, Khoa D. Doan},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
