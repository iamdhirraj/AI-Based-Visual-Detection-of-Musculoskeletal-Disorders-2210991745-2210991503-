# AI-Based Visual Detection of Musculoskeletal Disorders with Personalized Exercise Recommendation

**Authors:** Kapil Tanwar (Roll no 2210991745), Dheeraj Sharma (Roll no 2210991503)

## Abstract

This work presents a practical demonstration of automated musculoskeletal disorder detection from radiographs using a suite of deep learning models. We compare classical convolutional architectures (VGG, ResNet, DenseNet) and lightweight CNN baselines on MURA-derived statistics. The demo provides reproducible artifacts including model performance summaries, visualizations, and a corrected PDF report.

## Introduction

Musculoskeletal disorders are a common cause of pain and disability. Automated radiograph interpretation can assist clinicians by triaging studies and recommending personalized rehabilitation exercises. This project showcases an end-to-end demo using precomputed/synthetic metrics and visualization assets to illustrate typical research outcomes.

## Dataset

We used aggregated statistics derived from the Stanford MURA dataset: 14,863 studies across seven body parts. Classes are binary (normal vs abnormal) with slight class imbalance (≈61% normal, 39% abnormal).

## Methods

- Model families evaluated: CNN baselines, VGG16/19, ResNet50/152V2, DenseNet (frozen and fine-tuned).
- Metrics reported: AUC, Accuracy, Cohen's Kappa.
- Visualization: bar charts, donut/pie charts, and performance comparison plots generated with Matplotlib.

## Results (summary)

- DenseNet (Fine-tuned) achieved the best synthetic performance: AUC 0.93, Accuracy 0.87, Kappa 0.82.
- DenseNet (Frozen) and ResNet152V2 followed closely.
- Dataset distribution: Wrist (31.3%), Shoulder (19.4%), Finger (17.2%), other parts lower.

## Discussion

The demo is intended for presentation and reproducibility rather than novel model development. It demonstrates how to package experimental artifacts, generate a reproducible PDF report, and present results in a simple static website.

## Reproducibility / How to run

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Run `python demo.py` to regenerate PNG visualizations in `demo_output/`.
3. Run `python generate_pdf.py` to produce the corrected PDF report.
4. Open `index.html` to view the dashboard and demo results.

## Conclusion

This project provides an accessible demonstration of radiograph analysis workflows and packaging for academic presentation. The included assets and scripts are sufficient to reproduce the demo outputs on a standard workstation.

## References

1. Stanford MURA dataset: https://stanfordmlgroup.github.io/competitions/mura/
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
3. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks.
