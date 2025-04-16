# Text-Summarization-Using-LLMs
# For BERT:

## Chosen Loss Function: Cross-Entropy Loss
## Explanation of the Choice:
In your code, you are using BertForSequenceClassification for the task of extractive summarization, where sentences are classified as either relevant (summary, label=1) or not relevant (non-summary, label=0). The model outputs logits for each class, and the task is a binary classification problem.
The Cross-Entropy Loss is appropriate for this setup because:
1.	Classification Task: Cross-Entropy Loss is the standard choice for classification problems, as it measures the difference between the predicted class probabilities (after applying softmax) and the true labels. It penalizes incorrect predictions more heavily, helping the model learn better.
2.	Compatibility with BERT: The BertForSequenceClassification model uses Cross-Entropy Loss by default for training. This is implemented within the Trainer class provided by Hugging Face, so there is no explicit need to specify it unless custom loss computation is required.
3.	Binary Problem Representation: While the task involves binary labels (0 or 1), the model treats it as a multi-class problem with two classes. Cross-Entropy Loss can handle this setup efficiently by comparing the logits for the two classes with the true labels.
4.	Optimization Efficiency: Cross-Entropy Loss is differentiable and works well with gradient descent optimizers like Adam (used implicitly in the Trainer), making it suitable for deep learning models.
 
## Why Not Other Loss Functions?
●	Mean Squared Error (MSE): MSE is more suited for regression tasks. It would not perform well here since the output probabilities are not continuous values to be regressed but discrete class probabilities.
●	Hinge Loss: Hinge Loss is used in Support Vector Machines (SVMs) and is not ideal for probabilistic models like BERT.
●	Binary Cross-Entropy (BCE): BCE could also be used if the problem was treated as a single-output binary classification task. However, in multi-class classification with two labels, Cross-Entropy Loss is preferred.
 
## Evaluation Metrics
For summarization tasks, metrics like ROUGE-1, ROUGE-2, and ROUGE-L are excellent for evaluating how closely the generated summaries match the ground truth summaries. These metrics complement the loss by providing insights into how well the model performs in generating summaries that align with human-written ones.


 

 

 


 
![image](https://github.com/user-attachments/assets/511dca54-f0c0-484f-ab17-8dfdb44e8953)


 


 


#FOR GPT


If you're fine-tuning GPT-2 for extractive summarization, the choice of loss function and evaluation strategy should align with GPT-2's architecture and purpose as a generative model rather than a classification-focused model like BERT.
 
Loss Function for GPT-2
1.	Chosen Loss Function: Cross-Entropy Loss
○	Why?
■	GPT-2, being a generative model, predicts the next token in a sequence given the previous tokens. During fine-tuning, Cross-Entropy Loss is the standard choice as it measures the difference between the predicted token probabilities and the actual token labels.
■	For summarization tasks, you can frame the problem such that GPT-2 generates a sequence containing the most relevant sentences (summary). The loss is computed token by token over the generated sequence.
 
## Key Considerations for GPT-2 Fine-Tuning:
###1.	Input and Output Preparation:
○	GPT-2 requires an input sequence and a target sequence for training. For summarization:
■	Input: The original text (dialogue in your dataset).
■	Target: A sequence containing the extracted summary (sentences marked as relevant).
○	Use GPT-2's tokenizer to process the text and ensure the input and target sequences are aligned.
###2.	Padding and Attention Masking:
○	Use appropriate padding and attention masks to ensure that the model doesn't compute loss for padded tokens.
###3.	Implementation with Hugging Face:
○	When using the Trainer class, GPT-2 also uses Cross-Entropy Loss by default for token-level predictions. You don't need to specify it unless you’re implementing a custom training loop.
 
## Evaluation Metrics for GPT-2
In addition to loss, the following metrics are appropriate for evaluating a summarization task with GPT-2:
###1.	Perplexity:
○	Perplexity is commonly used for generative models like GPT-2. It measures how well the model predicts the target sequence and is a direct function of Cross-Entropy Loss.
###2.	ROUGE Scores:
○	As in BERT, ROUGE-1, ROUGE-2, and ROUGE-L can evaluate how well the generated summaries match the ground-truth summaries.
###3.	BLEU Score (Optional):
○	While less common in summarization tasks compared to ROUGE, BLEU can also measure overlap between the model’s output and reference summaries.


 
 

 

 
![image](https://github.com/user-attachments/assets/6799f870-a36b-4ee5-b4d7-019701435d94)

