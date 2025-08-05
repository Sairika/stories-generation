# Fine-Tune GPT-2 to Generate Stories

## Overview
This repository contains a Jupyter Notebook that demonstrates how to fine-tune the GPT-2 model to generate creative stories based on writing prompts. The notebook utilizes the Hugging Face Transformers library and is designed to run in a Kaggle Python environment with GPU support.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a diverse dataset of web pages. This project aims to fine-tune GPT-2 on a specific dataset of writing prompts and corresponding stories to improve its ability to generate coherent and creative narratives.

## Requirements
To run the notebook, you will need:
- Python 3.x
- PyTorch
- Hugging Face Transformers library (version 3.0.2 or later)
- Kaggle account (for running the notebook in a Kaggle environment)

### Installation
You can install the required libraries using the following commands:
```bash
!pip install transformers
```

## Dataset
The dataset used for fine-tuning consists of writing prompts and stories. The prompts and stories are stored in separate files, and the notebook combines them for training. The dataset can be downloaded from the original source mentioned in the notebook.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fine-tune-gpt-2-to-generate-stories.git
   cd fine-tune-gpt-2-to-generate-stories
   ```

2. Open the Jupyter Notebook in a Kaggle environment or your local setup.

3. Follow the instructions in the notebook to:
   - Load the dataset
   - Preprocess the data
   - Fine-tune the GPT-2 model
   - Generate stories based on prompts

## Results
After fine-tuning, the model's performance is evaluated using perplexity as a metric. The notebook includes examples of generated stories before and after fine-tuning, showcasing the improvements in coherence and creativity.

## Conclusion
Fine-tuning GPT-2 with a specific dataset can significantly enhance its storytelling capabilities. However, the complexity of human writing remains a challenge, and further exploration in this field is encouraged.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute to this project by submitting issues or pull requests. Happy coding!
