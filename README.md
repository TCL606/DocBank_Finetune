# DocBank

## Code

Code in docbank_ft is my code to get layoutlm baseline on DocBank dataset. Examples and Layoutlmft are code in [Microsoft Repo](https://github.com/microsoft/unilm).

## Installation

~~~bash
conda create -n layoutlmft python=3.7
conda activate layoutlmft
git clone https://github.com/microsoft/unilm.git
cd unilm
cd layoutlmft
pip install -r requirements.txt
pip install -e .
~~~

## Results

My results of Layoutlm on DocBank are as follows

|          | Abstract | Author | Caption | Date   | Equation | Figure | Footer | List   | Paragraph | Reference | Section | Table  |
| -------- | -------- | ------ | ------- | ------ | -------- | ------ | ------ | ------ | --------- | --------- | ------- | ------ |
| LayoutLM | 0.9857   | 0.9144 | 0.9611  | 0.8001 | 0.9106   | 1.0    | 0.9288 | 0.9076 | 0.9811    | 0.9378    | 0.9635  | 0.8674 |

