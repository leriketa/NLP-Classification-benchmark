# NLP-Classification-benchmark. Cost vs Performance
This project aims to analyze the trade-off between cost and performance when selecting the optimal model for NLP classification tasks. I built an NLP classification benchmark comparing traditional machine learning against modern deep learning on a sentiment analysis task to discover the most cost-effective method.


Dataset & Environment

• Dataset: stanfordnlp/imdb via Hugging Face (Movie reviews labeled as 0 for negative and 1 for positive).
• Environment: Google Colab (utilizing a free T4 GPU accelerator).
• Data Split: A balanced subset of 1,000 samples for training (500 positive / 500 negative) and 500 samples for testing (250 positive / 250 negative) to prevent the models from developing an architectural bias toward a single class.


Technical Implementation


1. Data Preprocessing & Cleaning

• HTML Cleaning: The raw dataset contained web line breaks. Since garbage text lowers final model scores, I used a Python string replacement pipeline to strip them out before tokenization.

• Tokenization: For the Transformer stack, text was processed using the pre-trained distilbert-base-uncased tokenizer from the Hugging Face transformers library to map text directly into numeric input IDs.


2. Deep Learning Pipeline (DistilBERT)

• Model Loading: Loaded AutoModelForSequenceClassification configured for two target labels.

• The "Missing/Unexpected" Warning: Upon loading the base weights, the library flagged missing head layers. This is normal; the model discarded its pre-trained fill-in-the-blank vocabulary tools to build a blank workspace (the classifier) dedicated to sentiment analysis.

• Tensors & Logits: Input IDs were packaged into Rank-2 Tensors [batch_size, sequence_length]. Passing them through the untrained model yielded unpolished raw scores (logits) close to zero, proving the initial model was guessing at random.

• The Padding Fix: During training initialization, the model glitched due to variations in review text lengths. I resolved this by integrating a DataCollatorWithPadding to dynamically apply placeholder padding tokens, standardizing batch shapes seamlessly.


3. Traditional Machine Learning Pipelines (Skikit-Learn)

• Feature Extraction: Leveraged TfidfVectorizer (max 5,000 features) to extract word-frequency matrices.

• Array Optimization: Utilized NumPy-backed array structures for ultra-fast arithmetic execution.

• Models: Evaluated a Logistic Regression classifier and a Support Vector Machine (LinearSVC) across identical balanced splits.

Benchmark Results


Model Architecture	          Test Accuracy	  F1-Score	    Training Time	Computational Footprint
TF-IDF + Logistic Regression	81.60%	0.82	 0.03 seconds 	Ultra-Lightweight (CPU)


TF-IDF + SVM	                80.80%	0.80	 0.02 seconds 	Ultra-Lightweight (CPU)


Transformer (DistilBERT)	    79.00%	0.80	 169.09 seconds Heavy (Requires GPU)


Key Engineering Takeaways

• The Winner: For this small slice of data (1,000 samples), traditional Logistic Regression beat the fancy Transformer in both absolute accuracy and inference speed.


• Pragmatic AI Architecture: Using a deep learning model is not always suited for every business task. This project proves that simpler, traditional models can deliver superior results instantly without incurring massive server cloud computing costs.


Project Limitations

• Data Scale: A sample size of 1,000 rows is highly restrictive. While DistilBERT is mathematically superior at context tracking, it is a data-hungry architecture. It requires a larger threshold (typically 10,000+ examples) to out-train a simple keyword-counting baseline.

• Hyperparameter Tuning: Models were trained using standard base configurations. Implementing a grid search over TF-IDF token constraints or stretching DistilBERT training epochs could shift subtle baseline variances.



