arXiv:2506.01808v1 [cs.CL] 2 Jun 2025
**NAVER LABS** Europe
June, 2025

# NAVER LABS Europe Submission to the Instruction-following Track

Beomseok Lee<sup>123★</sup>, Marcely Zanon Boito<sup>1★</sup>, Laurent Besacier<sup>1</sup>, Ioan Calapodescu<sup>1</sup>

<sup>1</sup>NAVER LABS Europe, France
<sup>2</sup>University of Trento, Italy
<sup>3</sup>Fondazione Bruno Kessler, Italy
<sup>★</sup>equal contribution



<table>
  <caption>Figure 1: Our training pipeline. A speech projector (A) and text LoRA adapters (B) are trained in parallel using speech-to-text and text-to-text data, respectively. These modules are then integrated during a brief multimodal adaptation step (C).</caption>
  <thead>
    <tr>
      <th>Feature / Component</th>
      <th>(A) Speech Projector</th>
      <th>(B) Text LoRA Adapters</th>
      <th>(C) Multimodal (A+B)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LLM</th>
      <td>Frozen weights</td>
      <td>Frozen weights</td>
      <td>Frozen weights</td>
    </tr>
    <tr>
      <th>Projector</th>
      <td>Trainable weights</td>
      <td>Not present</td>
      <td>Trainable weights</td>
    </tr>
    <tr>
      <th>SFM</th>
      <td>Frozen weights</td>
      <td>Not present</td>
      <td>Frozen weights</td>
    </tr>
    <tr>
      <th>LoRA Adapter</th>
      <td>Not present</td>
      <td>Trainable weights</td>
      <td>Trainable weights</td>
    </tr>
    <tr>
      <th>Speech modality data tasks</th>
      <td>ASR, ST, SQA</td>
      <td>Not present</td>
      <td>ASR, ST, SQA, fluent SQA</td>
    </tr>
    <tr>
      <th>Text modality data tasks</th>
      <td>Not present</td>
      <td>MT, QA</td>
      <td>MT, QA, fluent QA</td>
    </tr>
  </tbody>
</table>



Figure 1: Our training pipeline. A speech projector (A) and text LoRA adapters (B) are trained in parallel using speech-to-text and text-to-text data, respectively. These modules are then integrated during a brief multimodal adaptation step (C).

### Abstract

In this paper we describe NAVER LABS Europe submission to the instruction-following speech processing short track at IWSLT 2025. We participate in the constrained settings, developing systems that can simultaneously perform ASR, ST, and SQA tasks from English speech input into the following target languages: Chinese, Italian, and German. Our solution leverages two pretrained modules: (1) a speech-to-LLM embedding projector trained using representations from the `SeamlessM4T-v2-large` speech encoder; and (2) LoRA adapters trained on text data on top of `Llama-3.1-8B-Instruct`. These modules are jointly loaded and further instruction-tuned for 1K steps on multilingual and multimodal data to form our final system submitted for evaluation.

### 1. Introduction

Large language models (LLMs) have demonstrated remarkable success across various text-based natural language processing tasks [2, 3, 13, 19, 32, 34], motivating research into extending them to other modalities. This has led to the development of multimodal LLMs capable of processing speech, audio, images and video [4, 6, 7, 10, 11, 14, 17, 20, 26, 28, 29].

This year IWSLT Instruction-following Speech Processing Track focuses on the leveraging of LLMs and speech foundation models (SFM) to build solutions capable to perform multilingual tasks from English speech input  and textual multilingual instructions [1]. NAVER LABS Europe (NLE) participates in the constrained setting of the *short track*, where the tasks proposed are automatic speech recognition (ASR), speech translation (ST) and multilingual spoken question answering (SQA). The target languages for ST and multilingual SQA are Chinese, Italian and German. The participants are allowed to use the speech backbone `SeamlessM4T-v2-large` [5] and the text LLM `Llama-3.1-8B-Instruct` [8] for both training and data generation.

Our submitted systems leverage all the available data from the constrained settings, together with

Corresponding author: marcely.zanon-boito@naverlabs.com

NAVER LABS Europe Submission to the Instruction-following Track

data automatically obtained using both backbones. We train two types of systems in parallel: (1) speech-to-text ASR/ST/SQA projectors that project the averaged speech representation from the SeamlessM4T-v2-large encoder to the embedding space of a frozen Llama-3.1-8B-Instruct; (2) text-only LoRA adapters [9], plugged on top of the same frozen LLM. Once both systems are separately trained, we show that we can merge them, increasing overall speech performance, by fine-tuning for only 1K steps on multimodal multilingual data.

This system paper is organized as follows. Section 2 describes the preprocessing applied to the data used in this challenge. Sections 3 and 4 describe our training pipeline and experimental settings, respectively. Section 5 presents our experiments and discussion. Section 6 presents the submitted system. Section 7 concludes the paper.

# 2. Data

For training our models, we leverage the data from the constrained setting: CoVoST2 [33], EuroParlST [12] and SpokenSQuAD [15]. With the agreement of the organizers, we also take advantage of the SeamlessM4T-v2-large to produce extra synthetic speech data (Seamless TTS) and multilingual text data (Seamless MT). Llama-3.1-8B-Instruct is used to rephrase SQA answers. ACL 60-60 [27] is used for validation and evaluation only. We now present our data preprocessing (Section 2.1) and prompt format (Section 2.2).

## 2.1. Data Preprocessing

We produce both speech-to-text and text-to-text instructions to train our systems. For the aforementioned datasets, we produce the following splits, where * denotes synthetic splits obtained via SeamlessM4T-v2-large MT; † indicates splits generated with SeamlessM4T-v2-large TTS; and ‡ marks those derived through Llama-3.1-8B-Instruct-based rephrasing.

* **CoVoST2:**
    - ASR and ST/MT (en-de; en-zh)
* **EuroParlST:**
    - ASR and ST/MT (en-de; en-it)
* **SpokenSQuAD:**
    - ASR<sup>†</sup> and MT (en<sup>†</sup>-de*; en<sup>†</sup>-it*; en<sup>†</sup>-zh*)
    - SQA/QA (en<sup>†</sup>-en; en<sup>†</sup>-de*; en<sup>†</sup>-it*; en<sup>†</sup>-zh*)
    - fluent SQA/QA (en<sup>†</sup>-en<sup>‡</sup>; en<sup>†</sup>-de<sup>∗,‡</sup>; en<sup>†</sup>-it<sup>∗,‡</sup>; en<sup>†</sup>-zh<sup>∗,‡</sup>)
* **ACL 60-60:**
    - ASR and ST/MT (en-de; en-it*; en-zh)

Below we detail dataset-specific preprocessing. Statistics are presented in Table 1.

**CoVoST2 and EuroParlST** CoVoST2 covers English to German and Simplified Chinese language directions. EuroParlST covers English to German and Italian. ASR splits for these datasets were built by merging the existing language splits and deduplicating the audio files. For both, language-specific ST and MT splits are created by aligning translations to English speech and reference transcriptions, respectively.

**SpokenSQuAD** The SpokenSQuAD dataset is organized into two jsons, train and test. Each split is organized in themes, each with several paragraphs. For each paragraph, TTS audio files are available, aligned at the sentence level.<sup>1</sup> For each sentence, questions (each with several answers) are available. We performed the following modifications to this dataset:

* **Duplicated answers:** we removed duplicated answers to the same question using exact string matching, as well as any questions that required more than one audio file to answer, since we are participating in the SHORT track.
* **Validation set:** we created a validation set by selecting the first 20 themes of the training set and removing them from training (3,102 entries).
* **New TTS audio:** we generate new TTS audio files for the training set using SeamlessM4T-v2-large. Resynthesizing the source audio for this dataset was necessary due to existing dataset misalignment, which we detail in Appendix Section A.1.
* **Multilingual SQA/QA:** we created multilingual SQA/QA sets by first translating questions and answers to target languages using SeamlessM4T-v2-large. We then use reference-free COMET<sup>2</sup> [25] to filter out all pairs of questions and answers that do not both score at least 0.85.
* **Invalid splits:** we created the *invalid* SQA sets by deliberately mismatching context and question themes, thereby creating unanswerable examples. The corresponding answers were labeled as "Not answerable" in four languages (English, Italian, German and Chinese), following the guideline answer provided by the task organizers. While we acknowledge that a small, unknown subset of these reassigned questions may still be answerable, we hypothesize that ensuring a thematic mismatch

<sup>1</sup>The format is themeID_paragraphID_sentenceID.
<sup>2</sup>Unbabel/wmt22-cometkiwi-da

2

NAVER LABS Europe Submission to the Instruction-following Track

<table>
  <thead>
    <tr>
        <th>Dataset</th>
        <th>Task</th>
        <th>Language</th>
        <th># Samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td rowspan="3">CoVoST2</td>
        <td>ASR</td>
        <td>en</td>
        <td>289,413</td>
    </tr>
    <tr>
        <td rowspan="2">ST/MT</td>
        <td>en-de</td>
        <td>289,413</td>
    </tr>
    <tr>
        <td>en-zh</td>
        <td>289,413</td>
    </tr>
    <tr>
        <td rowspan="3">EuroParlST</td>
        <td>ASR</td>
        <td>en</td>
        <td>35,372</td>
    </tr>
    <tr>
        <td rowspan="2">ST/MT</td>
        <td>en-de</td>
        <td>32,628</td>
    </tr>
    <tr>
        <td>en-it</td>
        <td>29,552</td>
    </tr>
    <tr>
        <td rowspan="12">SpokenSQuAD</td>
        <td>ASR</td>
        <td>en</td>
        <td>34,003</td>
    </tr>
    <tr>
        <td rowspan="3">MT</td>
        <td>en-de</td>
        <td>39,362</td>
    </tr>
    <tr>
        <td>en-it</td>
        <td>55,030</td>
    </tr>
    <tr>
        <td>en-zh</td>
        <td>25,078</td>
    </tr>
    <tr>
        <td rowspan="4">SQA/QA</td>
        <td>en-en</td>
        <td>34,003†</td>
    </tr>
    <tr>
        <td>en-de</td>
        <td>6,574†</td>
    </tr>
    <tr>
        <td>en-it</td>
        <td>16,767†</td>
    </tr>
    <tr>
        <td>en-zh</td>
        <td>7,093†</td>
    </tr>
    <tr>
        <td rowspan="4">fluent SQA/QA</td>
        <td>en-en</td>
        <td>32,320</td>
    </tr>
    <tr>
        <td>en-de</td>
        <td>4,169</td>
    </tr>
    <tr>
        <td>en-it</td>
        <td>13,712</td>
    </tr>
    <tr>
        <td>en-zh</td>
        <td>3,424</td>
    </tr>
  </tbody>
</table>

Table 1: Training sets statistics by task. For ST/MT sets, target side is duplicated. For SpokenSQuAD, † highlights that the source speech is used twice (valid and invalid questions, as described in Section 2.1).

English speech as input. In this setting, the language of the question must match the language of the answer.

To develop a model capable of smoothly switching between different tasks, we designed task prompts with a consistent structure: regardless of the task (ASR, ST, or SQA), the user turn begins by encapsulating the speech embeddings within textual tags. This is followed on a new line by a task-specific instruction formulated as a question, and finally, another line containing a common suffix. The list of templates used is available in Appendix Table 4.

## 3. Training Pipeline

Our training pipeline is illustrated in Figure 1. We first train a speech projector on speech tasks (A), and text LoRA weights on textual tasks (B). These modules are then reloaded and adapted together on both speech and textual tasks (C). In this section we describe the key components used and the data sampling strategy.

between the reference context and the question is the most effective strategy for minimizing this issue.

* **Fluent SQA/QA version:** we created an alternative SQA/QA training set using Llama-3.1-8B-Instruct to regenerate the dataset original answers as fluent sentences. The motivation behind this was the observation that, since the original answers are made of an exact extract of the reference audio/text, the model had a difficult time answering some out-of-domain questions fluently.<sup>3</sup> More details are presented in Appendix Section A.2.

**Foundation Models** For speech, we leverage SeamlessM4T-v2-large model, extracting speech representations for all our audio data from its 24th speech encoder layer (i.e. the last layer). Prior to training, we average every 3 consecutive frame vectors, reducing significantly the sequence length. This simple trick allows us to train our models with larger batches, while maintaining good performance in speech tasks. All our models are built on top of a frozen Llama-3.1-8B-Instruct.

ACL 60-60 We use SeamlessM4T-v2-large to generate Italian translations, since the data shared only contained en-de and en-zh splits. We leverage the dev set for checkpoint selection during training. The eval set is used for testing.

**Speech Projector Architecture** The speech projector consists of 4 Transformer encoder layers, each with 8 attention heads. The input dimension is set to 1,024, the feed-forward network dimension to 2,048, and the output dimension to 4,096 to align with the embedding size of Llama-3.1-8B-Instruct. A dropout rate of 0.1 is applied throughout, and the model employs pre-layer normalization.

### 2.2. Prompt Format

The goal of the short track of this challenge is to produce a model that is capable to 1) transcribe English speech; 2) translate English speech into Italian, German and Chinese; 3) Answer multilingual questions using

**LoRA Adapters** LoRA adaptation [9] is applied to both the self-attention (Q/K values, output projection) and feed-forward modules, and across all LLM layers. We use $rank = 8, \alpha = 16$. We do not use dropout.

<sup>3</sup>In the official IWSLT 2025 test set we observed examples as the following. Question: “What are the names of the speakers?” Our model’s answer: “yin and my colleague jiang”. While the model answer is an exact and correct extraction from the audio, we were unsure about how this would be considered during evaluation.

**Data Sampling Strategy** For training all our models, we define an epoch as $X$ steps across the dataset, with $X = \frac{|speech\_examples|}{batch\_size}$. To construct the training data for each epoch, we sample batches by first applying the predefined task-level sampling ratios, followed by sampling based on the internal domain-level splits within each task. In the case of multimodal training (speech

3

NAVER LABS Europe Submission to the Instruction-following Track

and text tasks mixed), we consider speech as our *main* modality, using it for defining epoch size and task ratio. Each time we sample a task and language split that has a textual equivalent (e.g., ST corresponds to MT; SQA to QA), we also sample a batch from the corresponding textual task. In practice, this means that every time a batch from ST en-de split is sampled, a batch from MT en-de follows it. We hypothesize that interleaving similar speech and textual batches during training provides regularization benefits, with the text data serving as a stabilizing signal for learning [21].

## 4. Experimental Settings

**Codebase** We train our models using an internal fork of torchtune [31], which allows us to process interleaved representations of text and high-dimensional vectors within the user turn during instruction tuning. The high-dimensional vectors pass through our speech projector, while the text prefix and suffix user prompts are processed by the LLM embedding layer. The obtained speech and text embeddings are both concatenated and fed into the first layer of the LLM which is trained on the masked input with standard cross-entropy loss. Different learning rate schedulers and optimizers are employed for the speech projector and the LoRA weights, allowing for more controlled and effective training of these distinct model components.

**Inference Settings** We perform inference using torchtune, with a batch size of 1 and greedy decoding. The maximum number of new tokens was limited to 300. Unless explicitly stated otherwise, this decoding strategy was consistently applied across all experimental settings. Additional discussion regarding multimodal inference is present in Appendix Section B.1.

**Evaluation Metrics** We evaluate our models on speech (ASR, ST, SQA) and text (MT, QA) tasks when relevant. For ASR, we score word error rate (WER) using HuggingFace evaluate library with default settings and MMS normalization [23]. For ST/MT we present two evaluation metrics: BLEU4 computed with sacrebleu library [22]<sup>4</sup>, and COMET [25].<sup>5</sup> For SQA/QA, we use LLM-as-a-judge evaluation scripts from the bergen library<sup>6</sup> [24]. We use their “yes/no” quality assessment evaluation format including the English reference text, the multilingual questions and the generated answers. We report average accuracy

across four LLMs: EuroLLM-9B-Instruct [18], Gemma3-12B/27B-Instruct [30], and Llama3.1-70B-Instruct.

**Baselines** We compare our results with both backbones we use for training. We evaluate MT and QA using the reference transcripts and Llama-3.1-8B-Instruct in zero-shot settings, and we evaluate SeamlessM4T-v2-large in both ASR and ST.

## 5. Experiments

We now present our results for ASR, ST, and SQA. Section 5.1 introduces the models used in our experiments, followed by results and discussion in Section 5.2.

### 5.1. Our Models

In this section, we describe the models used in our experiments. Additional hyperparameter details are provided in Appendix C.

**A.1 Speech Projector (ASR/ST)** This version of our speech projector focuses on ASR and ST tasks, which do not require any complex reasoning capability of the LLM. For ASR, we sample CoVoST2 and EuroParlST with probabilities 0.8 and 0.2 respectively. For ST, we sample proportionally to the datasets size, and set language sampling probabilities for the pairs en-de, en-zh, and en-it to 0.3, 0.4, and 0.3 respectively. We trained for 4 epochs using AdamW with learning rate of $1e-4$, a constant learning rate scheduler, and gradient accumulation of 16. This model trains for 4.6 days in a single A100-80GB, and the best checkpoint is obtained after 18.2 hours.
**A.1 Speech Projector (ASR/ST)** This version of our speech projector focuses on ASR and ST tasks, which do not require any complex reasoning capability of the LLM. For ASR, we sample CoVoST2 and EuroParlST with probabilities 0.8 and 0.2 respectively. For ST, we sample proportionally to the datasets size, and set language sampling probabilities for the pairs en-de, en-zh, and en-it to 0.3, 0.4, and 0.3 respectively. We trained for 4 epochs using AdamW with learning rate of $1e^{-4}$, a constant learning rate scheduler, and gradient accumulation of 16. This model trains for 4.6 days in a single A100-80GB, and the best checkpoint is obtained after 18.2 hours.
**A.2 Speech Projector (ASR/ST/SQA)** This version of our projector extends A.1, including the SQA task. We use task sampling probabilities of 0.4, 0.35, and 0.25 for ASR, ST and SQA respectively. For the ASR and ST tasks, we use the same data ratios as defined above. For the SQA task, we leveraged both valid and invalid splits, with language-specific sampling probabilities for English, German, and Italian set to 0.4, 0.3 and 0.3, respectively. We do not train with Chinese SQA. This model trains for 4.75 days in a single A100-80GB, and the best checkpoint is obtained after 27.36 hours. The best checkpoints for both versions of the speech projector are selected using its average ST performance on the ACL 60-60 dev split across all language directions. Additional hyper-parameters for A.1 and A.2 are presented in Appendix Section C.1.

<sup>4</sup>The signature is “nrefs:1|case:mixed|eff:no|tok:TOK|smooth:exp|version:2.3.1”, with *TOK = zh* for Chinese, and
<sup>4</sup>The signature is “nrefs:1|case:mixed|eff:no|tok:TOK|smooth:exp|version:2.3.1”, with *TOK = zh* for Chinese, and
13 for the other languages.
13a for the other languages.
<sup>5</sup>Unbabel/wmt22-comet-da

<sup>6</sup>https://github.com/naver/bergen

4

NAVER LABS Europe Submission to the Instruction-following Track

**B. Text-only LoRA (MT/QA)** We train LoRA weights on top of `Llama-3.1-8B-Instruct` using all text-to-text data from Table 1, and by using probability sampling of 0.6 and 0.4 for MT and QA respectively. We train for one epoch using AdamW with learning rate of $3e-4$, weight decay of 0.1, and 100 warm-up steps. Batch size of 10, and gradient accumulation of 8 is used. This model trains for approximately 4 days in a single A100-80GB. We select the last checkpoint.

**C. Multimodal (A.x + B)** We restart training by using both one of the speech projectors detailed above, and the text-only LoRA weights. We adapt our models using all speech (ASR/ST/SQA) and textual (MT/QA) tasks. We experiment with two versions of the SQA/QA training sets: the original short lowercase answers, and the *fluent* SQA/QA version we created. In preliminary experiments, we observed that as little as 100 steps were enough to successfully integrate the projector representation to the LoRA weights, but the best performance gains were obtained with 1K steps, which is the value we adopt for the experiments presented in the next section. We use learning rate of $1e-5$ for the speech projector, and of $3e-4$ for the LoRA weights. We use a batch size of 16, and gradient accumulation of 16. This model trains for approximately 6 hours in a single A100-80GB. We select the last checkpoint.

## 5.2. Results and Discussion

Table 2 presents our results for ASR, ST/MT and SQA/QA. ACL 60-60 *eval set* is used for ASR and ST/MT. SpokenSQuAD official test set is used for English SQA/QA. A smaller automatically obtained version is used for multilingual SQA/QA.<sup>7</sup> In the top portion, we present results for `Llama-3.1-8B-Instruct` before and after LoRA fine-tuning on text-only data. The middle portion of the table presents the speech backbone (`SeamlessM4T-v2-large`) and our projector-only models: for these rows, the only adaptation is the training of a speech projector that is plugged to a frozen `Llama-3.1-8B-Instruct`. Finally, the bottom portion of the table presents results from the merging of our text backbone (text-only LoRA) and the projectors of the middle portion via multimodal training. Additionally, ACL 60-60 ASR/ST dev results are presented in Appendix Table 7.

**Performance of Text-only Models (topline)** We observe that zero-shot `Llama-3.1-8B-Instruct` presents strong performance in both MT and QA tasks. By adding LoRA adapters on top of it, we increase trans-

lation performance in detriment of QA performance. We partially attribute this drop in QA performance to the SpokenSQuAD answer format, that is very short and might be judged as incomplete by the LLM evaluation. However, we also scored ROUGE1 [16] recall, which measures the intersection between the reference answer tokens and the produced one, finding that those scores were similar to the LLM-as-a-judge metric.<sup>8</sup> This result confirms that the QA performance is worse after text adaptation.

**Speech Projectors** We observe that both A.1 and A.2 are equally capable of performing ASR and ST, which is consistent with the fact that both are trained on the same data. However, we observe that A.2, the model that trains with SQA data, is unable to produce SQA output. We believe this is a limitation of the projector approach: we train a model capable of biasing the output of the LLM, which works well for content tasks such as ASR and ST. For a reasoning task, further adaptation might be required in order to force the model to comply to the instruction. Additional results for CoVoST2 and EuroParlST are presented in Appendix Table 6, and they confirm that both models are very similar in ST performance.

**Multimodal Training** We observe that our efficient multimodal training is beneficial, consistently increasing scores for ASR and ST. Our multimodal models outperform the speech projector models by 1-2 WER points in the ASR task. For ST task, in BLEU scores, the multimodal models always outperform speech projector models while in COMET there are some mixed results, with some models presenting slight deterioration for some language pairs. Finally, in the SQA task, while the speech projector model A.2 failed to learn the task effectively, our multimodal models achieved strong results, outperforming the text topline (B) performance across all language settings, and reaching scores that are close to the `Llama-3.1-8B-Instruct` topline, despite working from the speech signal.

**ASR Performance** We observe that our models are competitive with `SeamlessM4T-v2-large`, scoring slightly worse than the baseline for some configurations. Overall, for ACL 60-60 we observe quite elevated WER scores, compared to the ones we obtained for the training ASR datasets (see Appendix Table 6). We believe this is partially due to the nature of the dataset. We manually inspected some of the data, observing that some

<sup>7</sup>Statistics for the multilingual test set are presented in Appendix Table 5.

<sup>8</sup>For en, de, it and zh splits ROUGE1 recall scores were respectively: 81.4%, 63.1%, 69.5%, 79.0%.

5

NAVER LABS Europe Submission to the Instruction-following Track

<table>
  <thead>
    <tr>
        <th rowspan="2">Model (fine-tuning tasks)</th>
        <th rowspan="2">ASR (WER)<br/>en</th>
        <th colspan="3">ST/MT (BLEU)</th>
        <th colspan="3">ST/MT (COMET)</th>
        <th colspan="4">SQA/QA (LLM-AS-A-JUDGE)</th>
    </tr>
    <tr>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
        <th>en-en</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="12">Text-only Models (MT/QA)</td>
    </tr>
    <tr>
        <td>Llama-3.1-8B-Instruct (zero-shot)</td>
        <td>-</td>
        <td>23.88</td>
        <td>35.51</td>
        <td>45.89</td>
        <td>0.779</td>
        <td>0.806</td>
        <td>0.809</td>
        <td>91.8%</td>
        <td>92.0%</td>
        <td>88.6%</td>
        <td>84.6%</td>
    </tr>
    <tr>
        <td>B. Text-only LoRA (MT/QA)</td>
        <td>-</td>
        <td>41.69</td>
        <td>48.31</td>
        <td>53.65</td>
        <td>0.838</td>
        <td>0.863</td>
        <td>0.867</td>
        <td>83.4%</td>
        <td>75.7%</td>
        <td>71.4%</td>
        <td>69.5%</td>
    </tr>
    <tr>
        <td colspan="12">Speech-only Models (ASR/ST/SQA)</td>
    </tr>
    <tr>
        <td>SeamlessM4T-v2-large</td>
        <td>17.6</td>
        <td>27.95</td>
        <td>43.54</td>
        <td>33.58</td>
        <td>0.737</td>
        <td>0.788</td>
        <td>0.753</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>A.1 Speech Projector (ASR/ST)</td>
        <td>19.8</td>
        <td>27.58</td>
        <td>36.30</td>
        <td>40.62</td>
        <td>0.760</td>
        <td>0.796</td>
        <td>0.793</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>A.2 Speech Projector (ASR/ST/SQA)</td>
        <td>19.9</td>
        <td>27.20</td>
        <td>36.60</td>
        <td>40.72</td>
        <td>0.760</td>
        <td>0.797</td>
        <td>0.792</td>
        <td>0.7%</td>
        <td>0.5%</td>
        <td>0.3%</td>
        <td>0.6%</td>
    </tr>
    <tr>
        <td colspan="12">Multimodal Models (ASR/ST/SQA)</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/MT/SQA/QA)</td>
        <td>17.7</td>
        <td>30.37</td>
        <td>41.22</td>
        <td>42.76</td>
        <td>0.758</td>
        <td>0.791</td>
        <td>0.795</td>
        <td>79.8%</td>
        <td>71.9%</td>
        <td>69.4%</td>
        <td>65.5%</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>18.6</td>
        <td>30.75</td>
        <td>40.48</td>
        <td>42.51</td>
        <td>0.755</td>
        <td>0.788</td>
        <td>0.789</td>
        <td>90.3%</td>
        <td>85.2%</td>
        <td>82.9%</td>
        <td>76.4%</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/SQA/QA)</td>
        <td>18.2</td>
        <td>29.91</td>
        <td>38.13</td>
        <td>43.12</td>
        <td>0.759</td>
        <td>0.786</td>
        <td>0.799</td>
        <td>80.5%</td>
        <td>74.9%</td>
        <td>68.0%</td>
        <td>66.7%</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>18.7</td>
        <td>29.68</td>
        <td>32.28</td>
        <td>43.38</td>
        <td>0.763</td>
        <td>0.782</td>
        <td>0.798</td>
        <td>91.1%</td>
        <td>87.3%</td>
        <td>84.8%</td>
        <td>78.0%</td>
    </tr>
  </tbody>
</table>

Table 2: Results for the different models and backbones used in this work. ASR and ST scores are obtained using ACL 60-60 eval set, while SQA/QA scores are obtained using SpokenSQuAD test set.

of the audio files contain *leftover fragments* from previous sentences. Moreover, looking at the transcriptions, we observed that these faithfully reproduced the audio, without applying any normalization or removing disfluencies (e.g. repeated and filler words were kept). We find that disfluent transcriptions are hardly produced by LLM-based models or `SeamlessM4T-v2-large`, that both tend to translate the content into a clean format. Therefore, we believe the WER scores presented in our results table are not representative of the models real ASR capabilities.

increase, we attribute this to some formatting bias that could be happening during adaptation. The Appendix Section C.2 discusses the matter further.

**German MT/ST Performance** Across all experiments, we observed that our models consistently performed poorly on German, the language for which we had the largest amount of training data. This could be attributed to the LLM's inherent limitations in handling German, as reflected by its relatively lower performance in zero-shot settings (`Llama-3.1-8B-Instruct`) for the en-de pair for both BLEU and COMET, compared to en-it and en-zh. Nonetheless, training the model on multilingual speech-only data for the ST task led to an improvement of 3–4 BLEU points. Incorporating multimodal data (i.e., both ST and MT) yielded an additional gain of 2–3 BLEU points, further enhancing performance. For COMET, speech-only and multimodal training does not improve COMET scores over the text topline.

**SQA performance** Overall, we observe that by replacing SQA by *fluent* SQA, we drastically increased our SQA/QA scores. However, results for ASR slightly deteriorate. We hypothesize that this is due to the old SQA task being closer to the ASR task. For the original SQA, the answer is always a direct transcript of a portion of the input text, which as a task has a better synergy with ASR. In intermediate experiments, we observed that adding the original SQA data to the multimodal training was always beneficial for the ASR performance of the model.

**Overall ST Performance** We observe that the speech projectors (A.1 and A.2) outperform the backbone `SeamlessM4T-v2-large` for en-zh using BLEU, and in all languages using COMET. Then, multimodal training further increases the BLEU scores, but in some cases, this training slightly hurts the COMET scores. Since this difference in COMET is very small (en-de 0.005; en-it 0.008; en-zh 0.004), and since the BLEU scores

**Final Discussion** Overall our results show that it is possible to train text (B) and speech adaptation (A) in parallel, and then to align both via joint instruction tuning (C). By separating the pretraining of both components, we are able to focus hyper-parameter search at the merging stage, using two components that are already competent in their respective modalities. Despite improvements in scores over speech-only models, our best models do not beat the topline working from text for ST, but they do outperform `SeamlessM4T-v2-large` across all language pairs and metrics. Moreover, for SQA, we highlight that the obtained scores are in some cases very close (en-en, en-it) to the text topline, despite using speech as input context.

# 6. Submitted Model

Table 2 presented the results for our multimodal models. Due to the reasons highlighted in Section 2.1, we only consider two models for submission: the ones which

6

NAVER LABS Europe Submission to the Instruction-following Track

were trained with *fluent* SQA. This is because we believe that these models will suffer the least from domain shift, since they are capable of producing full fluent sentences for SQA.

We observe that both models (A.1 and A.2-based) seem to be equally capable of ASR (18.6 and 18.7). We evaluated language confusion for these two splits, finding that the output produced was English in 98% and 98.5% of the cases respectively. These models differ more in terms of ST performance: they obtained averages of 37.9 and 35.1 BLEU score points respectively. Finally, these models present the following average accuracy for SQA: 83.7% and 85.3%. In summary, while the A.1-based model seem much more capable in ST, the A.2-based model has a very slight edge in ASR and SQA.

Therefore, we decided to select the A.1-based model as our primary submission model. For producing the decoding of the test set, we transform the instructions into our own prompt format, and submit the output of the same model, with the same decoding settings for all splits: greedy decoding using 150 as maximum number of tokens. A brief post-submission discussion is provided in Appendix Section C.2.1.

## 7. Conclusion

In this paper, we presented NLE’s submission to the instruction-following speech processing short track at IWSLT 2025, constrained setting. We developed multimodal models that simultaneously performed ASR, ST, and SQA tasks from English speech input into Chinese, German and Italian. Our approach is simple yet effective: we decoupled training, training a speech projector on speech-to-text tasks, and LoRA adapters on text-to-text tasks. Then, both modules are loaded and the resulting multimodal model was instruction-tuned for a few steps on multilingual and multimodal data to produce the final system submitted for evaluation.

### Acknowledgments

We thank our NLE colleagues Vassilina Nikoulina, for the implementation of the LLM-as-judge scripts using bergen, and Christian Wolf, for kindly verifying some of our German ST/SQA outputs. This work was partially funded by the European Horizon 2022 project UTTER (Unified Transcription and Translation for Extended Reality), under grant agreement No 101070631.

### References

[1] Idris Abdulmumin, Victor Agostinelli, Tanel Alumäe, Antonios Anastasopoulos, Ashwin, Luisa Bentivogli, Ondřej Bojar, Claudia Borg, Fethi Bougares, Roldano Cattoni, Mauro Cettolo, Lizhong Chen, William Chen, Raj Dabre, Yannick Estève, Marcello Federico, Marco Gaido, Dávid Javorský, Marek Kasztelnik, Tsz Kin Lam, Danni Liu, Evgeny Matusov, Chandresh Kumar Maurya, John P. McCrae, Salima Mdhaffar, Yasmin Moslem, Kenton Murray, Satoshi Nakamura, Matteo Negri, Jan Niehues, Atul Kr. Ojha, John E. Ortega, Sara Papi, Pavel Pecina, Peter Polák, Piotr Połeć, Beatrice Savoldi, Nivedita Sethiya, Claytone Sikasote, Matthias Sperber, Sebastian Stüker, Katsuhito Sudoh, Brian Thompson, Marco Turchi, Alex Waibel, Patrick Wilken, Rodolfo Zevallos, Vilém Zouhar, and Maike Züfle. Findings of the iwslt 2025 evaluation campaign. In *Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)*, Vienna, Austria (in-person and online), 2025. Association for Computational Linguistics. To appear. 1, 10

[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. 1

[3] Duarte Miguel Alves, José Pombal, Nuno M Guerreiro, Pedro Henrique Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, Pierre Colombo, José G. C. de Souza, and Andre Martins. Tower: An open multilingual large language model for translation-related tasks. In *First Conference on Language Modeling*, 2024. 1

[4] Kshitij Ambilduke, Ben Peters, Sonal Sannigrahi, Anil Keshwani, Tsz Kin Lam, Bruno Martins, Marcely Zanon Boito, and André FT Martins. From tower to spire: Adding the speech modality to a text-only llm. *arXiv preprint arXiv:2503.10620*, 2025. 1

[5] Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, et al. Seamlessm4t: Massively multilingual & multimodal machine translation. *arXiv preprint arXiv:2308.11596*, 2023. 1

[6] Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. *arXiv preprint arXiv:2410.00037*, 2024. 1

[7] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. *arXiv preprint arXiv:2303.03378*, 2023. 1

[8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024. 1

[9] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language

7

NAVER LABS Europe Submission to the Instruction-following Track

models. In *International Conference on Learning Representations*, 2022. 2, 3

[10] Shujie Hu, Long Zhou, Shujie Liu, Sanyuan Chen, Lingwei Meng, Hongkun Hao, Jing Pan, Xunying Liu, Jinyu Li, Sunit Sivasankaran, et al. Wavllm: Towards robust and adaptive speech large language model. *arXiv preprint arXiv:2404.00656*, 2024. 1

[11] Rongjie Huang, Mingze Li, Dongchao Yang, Jiatong Shi, Xuankai Chang, Zhenhui Ye, Yuning Wu, Zhiqing Hong, Jiawei Huang, Jinglin Liu, et al. Audiogpt: Understanding and generating speech, music, sound, and talking head. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 23802–23804, 2024. 1

[12] J. Iranzo-Sánchez, J. A. Silvestre-Cerdà, J. Jorge, N. Roselló, A. Giménez, A. Sanchis, J. Civera, and A. Juan. Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 8229–8233, 2020. 2

[13] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024. 1

[14] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. arXiv:2405.02246. 1

[15] Chia-Hsuan Lee, Szu-Lin Wu, Chi-Liang Liu, and Hung-yi Lee. Spoken squad: A study of mitigating the impact of speech recognition errors on listening comprehension. *Proc. Interspeech 2018*, pages 3459–3463, 2018. 2

[16] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain, 2004. Association for Computational Linguistics. 5

[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning (LLaVA), 2023. arXiv:2304.08485 [cs]. 1

[18] Pedro Henrique Martins, Patrick Fernandes, João Alves, Nuno M. Guerreiro, Ricardo Rei, Duarte M. Alves, José Pombal, Amin Farajian, Manuel Faysse, Mateusz Klimaszewski, Pierre Colombo, Barry Haddow, José G. C. de Souza, Alexandra Birch, and André F. T. Martins. Eurollm: Multilingual language models for europe, 2024. 4

[19] Pedro Henrique Martins, Patrick Fernandes, João Alves, Nuno M Guerreiro, Ricardo Rei, Duarte M Alves, José Pombal, Amin Farajian, Manuel Faysse, Mateusz Klimaszewski, et al. Eurollm: Multilingual language models for europe. *arXiv preprint arXiv:2409.16235*, 2024. 1

[20] Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R Costa-Jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, et al. Spirit-lm: Interleaved spoken and written language model. *Transactions of the Association for Computational Linguistics*, 13:30–52, 2025. 1

[21] Iñigo Pikabea, Iñaki Lacunza, Oriol Pareras, Carlos Escolano, Aitor Gonzalez-Agirre, Javier Hernando, and Marta Villegas. Breaking language barriers in visual language models via multilingual textual regularization. *arXiv preprint arXiv:2503.22577*, 2025. 4

[22] Matt Post. A call for clarity in reporting BLEU scores. In *Proceedings of the Third Conference on Machine Translation: Research Papers*, pages 186–191, Belgium, Brussels, 2018. Association for Computational Linguistics. 4

[23] Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, et al. Scaling speech technology to 1,000+ languages. *Journal of Machine Learning Research*, 25(97):1–52, 2024. 4

[24] David Rau, Hervé Déjean, Nadezhda Chirkova, Thibault Formal, Shuai Wang, Stéphane Clinchant, and Vassilina Nikoulina. BERGEN: A benchmarking library for retrieval-augmented generation. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 7640–7663, Miami, Florida, USA, 2024. Association for Computational Linguistics. 4

[25] Ricardo Rei, José G. C. de Souza, Duarte Alves, Chrysoula Zerva, Ana C Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, and André F. T. Martins. COMET-22: Unbabel-IST 2022 submission for the metrics shared task. In *Proceedings of the Seventh Conference on Machine Translation (WMT)*, pages 578–585, Abu Dhabi, United Arab Emirates (Hybrid), 2022. Association for Computational Linguistics. 2, 4

[26] Paul K Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, et al. Audiopalm: A large language model that can speak and listen. *arXiv preprint arXiv:2306.12925*, 2023. 1

[27] Elizabeth Salesky, Kareem Darwish, Mohamed Al-Badrashiny, Mona Diab, and Jan Niehues. Evaluating multilingual speech translation under realistic conditions with resegmentation and terminology. In *Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023)*, pages 62–78, Toronto, Canada (in-person and online), 2023. Association for Computational Linguistics. 2

[28] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn: Towards generic hearing abilities for large language models. *arXiv preprint arXiv:2310.13289*, 2023. 1

[29] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023. 1

[30] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey

8

NAVER LABS Europe Submission to the Instruction-following Track

Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucińska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin,

Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report, 2025. 4

[31] torchtune maintainers and contributors. torchtune: Pytorch’s finetuning library, 2024. 4

[32] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. 1

[33] Changhan Wang, Anne Wu, and Juan Pino. Covost 2: A massively multilingual speech-to-text translation corpus, 2020. 2

[34] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 1

9

NAVER LABS Europe Submission to the Instruction-following Track

# A. Data Preprocessing and Prompts

## A.1. SpokenSQuAD TTS data

In this section we explain the misalignment we found in the SpokenSQuAD train split, as well as our procedure for creating the new TTS split.

**Train Split Misalignment** During the preprocessing of the train split of SpokenSQuAD, we witnessed cases of misalignment: for some given paragraphs, the corresponding audios were shifted by a factor varying between 1 and 3. For instance, the first audio in a given paragraph was incorrectly named as “1”, instead of “0”, shifting all the paragraph’s alignment. We listened to all cases we were able to flag, manually correcting them. However, we believe this hinted to a deeper alignment issue, as the obtained training set seemed to be difficult to learn. We observed that models trained with this training set included were unable to generalize to SpokenSQuAD’s validation and test sets, always producing random Wikipedia sentences when receiving the SpokenSQuAD’s TTS voice as input.

**New TTS Source Audio Generation** We use SeamlessM4T-v2-large to produce new source audio for the SpokenSQuAD training set (34,003 sentences). For each entry in this set, we re-synthesize its SQuAD reference text by randomly sampling one of the 200 speakers present in SeamlessM4T. This results in a more diverse training set, since the original TTS used a single female voice for all sentences. We also generated extra speech data using all different questions present in this training set, producing a second ASR set containing speech for 28,000 questions.

## A.2. SpokenSQuAD Answers Regeneration

SpokenSQuAD answers are direct extracts from the reference text, formatted as lowercase text without punctuation. We discovered this presented a limitation when training our models on SQA. The trained models overfitted to that format, solving SQA as a transcription task of the relevant portion of the source audio. While conceptually correct, this approach can result in generalization issues if more than one extract is required to answer the question, as the model never observed such a setting during training, and it will thus have the tendency to transcribe everything between the two points of interest.

We use Llama-3.1-8B-Instruct to regenerate all answers in our multilingual training set. The prompt used for regeneration is presented in Table 3. After regeneration, we remove answers generated in the wrong language using an automatic language identification  tool. Statistics are presented in Table 1.

## A.3. Data Statistics

Table 5 presents the statistics for the multilingual test set of SpokenSQuAD.

## A.4. Our Prompts

Table 4 presents our prompt format. We designed our prompts to be very similar, independently of the target task. The language of the question defines the answer’s target language.

# B. Additional Results

**CoVoST2 and EuroParlST Results** Table 6 presents results for relevant models on the in-domain test sets from CoVoST2 and EuroParlST. We observe that the multimodal adaptation improves the speech projectors’ ASR and BLEU scores, while slightly decreasing COMET scores.

**ACL 60-60 Dev Results** Table 7 presents ACL 60-60 dev split scores for some of the models presented in the main results table (Table 2).

**SQA/QA BERT Scores** Table 8 presents BERT scores for the multimodal models presented in the main results table (Table 2), computed after the evaluation period and using the same settings from Abdulmumin et al. [1]. We observe that scores for languages other than English over the valid test set are considerably lower than the LLM-as-judge scores in Table 2. Prior to the evaluation period we had evaluated our models using BERT score and xlm-roberta-large, which yielded much higher scores, similar to those obtained in the LLM-as-judge evaluation. Those scores are presented in Table 9.

**Invalid Questions** In Table 8 we also present BERT scores for an invalid multilingual test set, made of the same reference audio files from the English test split (8,026 examples), but with incorrect questions. We observe that the BERT scores for this invalid split is very high, showcasing that our models are fully capable of respecting the instruction format for incorrect answers.

## B.1. Inference Issues

Through manual inspection of our model outputs, we observed that in a small number of cases, inference degenerates, resulting in repeated words or sentences until the maximum token limit is reached. We experimented with various inference strategies, greedy decoding, top-p, and top-k sampling, as well as different

10

NAVER LABS Europe Submission to the Instruction-following Track

**Context**: [REFERENCE TEXT]
**Question**: [QUESTION]
**Answer**: [ANSWER]
**Instruction**: Reformulate the answer to be a natural sounding sentence that answers the question in the correct language. Produce text in the same language of the question and answer. Do not make it too long, or add too much information. Don’t add anything else to your answer.

Table 3: Regeneration prompt we gave to Llama-3.1-8B-Instruct to regenerate the answers in the training set of SpokenSQuAD.


<table>
  <thead>
    <tr>
        <th colspan="2">User Prompt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>Speech Prefix</td>
        <td>Content: &lt;speech&gt;[SPEECH EMBEDDINGS]&lt;/speech&gt;\n</td>
    </tr>
    <tr>
        <td>Text Prefix</td>
        <td>Content: &lt;text&gt;[SPEECH TRANSCRIPTION]&lt;/text&gt;\n</td>
    </tr>
    <tr>
        <td>ASR</td>
        <td>Question: Can you transcribe the Speech content into English text?\n</td>
    </tr>
    <tr>
        <td>ST/MT (de)</td>
        <td>Question: Können Sie den Inhalt der Rede in den deutschen Text übersetzen?\n</td>
    </tr>
    <tr>
        <td>ST/MT (it)</td>
        <td>Question: Puoi tradurre il contenuto del discorso in testo italiano?\n</td>
    </tr>
    <tr>
        <td>ST/MT (zh)</td>
        <td>Question: 你能把演讲内容翻译成中文吗?\n</td>
    </tr>
    <tr>
        <td>SQA/QA</td>
        <td>Question: [QUESTION]\n</td>
    </tr>
    <tr>
        <td>Suffix</td>
        <td>Your answer:</td>
    </tr>
  </tbody>
</table>

Table 4: The user turn prompt template used for training our models. For speech tasks, the user prompt is given by Speech Prefix+Task+Suffix, for text tasks, the user prompt is given by Text Prefix+Task+Suffix. ST/MT instructions were obtained by translating the instruction “Can you translate the Speech content into [German/Italian/Chinese] text?” to corresponding target languages using SeamlessM4T-v2-large.


<table>
  <thead>
    <tr>
        <th> </th>
        <th>en</th>
        <th>de</th>
        <th>it</th>
        <th>zh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td># lines</td>
        <td>8,026</td>
        <td>3,272</td>
        <td>783</td>
        <td>627</td>
    </tr>
  </tbody>
</table>

wenn wir nach der kompositionalen Generalisierung testen, könnte es mir so vorkommen, als ob wir in diesem Fall die Mädchen schlafen, <mark>und ich sehe, dass sie schlafen, und ich sehe, dass sie schlafen, (..) und ich sehe, dass sie schlafen, und ich sehe, dass”</mark>
*wenn wir nach der kompositionalen Generalisierung testen, könnte es mir so vorkommen, als ob wir in diesem Fall die Mädchen schlafen, <span style="color: red">und ich sehe, dass sie schlafen, und ich sehe, dass sie schlafen, (..) und ich sehe, dass sie schlafen, und ich sehe, dass”</span>*
*wenn wir nach der kompositionalen Generalisierung testen, könnte es mir so vorkommen, als ob wir in diesem Fall die Mädchen schlafen, <mark>und ich sehe, dass sie schlafen, und ich sehe, dass sie schlafen, (..) und ich sehe, dass sie schlafen, und ich sehe, dass”</mark>*
Table 5: Statistics for the SpokenSQuAD SQA/QA test sets. The multilingual version is obtained via SeamlessM4T-v2-large translation of questions and answers, with posterior quality filtering based on COMET scores.

temperature settings, but were unable to identify a configuration that fully eliminated the issue. We hypothesize that a lightweight post-processing model could offer a simple and effective solution to mitigate this problem. Below we give some examples of inference degeneration for Chinese and German.

An example of inference degeneration in Chinese:

“国家语言理解模型从各种知识来源中提取出来，例如，<mark>通常通过预训练获得的参数中包含的知识，通常通过预训练获得的知识，通常通过预训练获得的知识，通常通过预训练获得的知识，(..)通常通过预训练练”</mark>
“国家语言理解模型从各种知识来源中提取出来，例如，<span style="color: red">通常通过预训练获得的参数中包含的知识，通常通过预训练获得的知识，通常通过预训练获得的知识，通常通过预训练获得的知识，(..)通常通过预训练”</span>
An example of inference degeneration in German:
“In Zusammenhang mit der semantischen Parsierung,

11

NAVER LABS Europe Submission to the Instruction-following Track

<table>
  <thead>
    <tr>
        <th rowspan="4"> </th>
        <th colspan="4">ASR</th>
        <th colspan="8">ST</th>
    </tr>
    <tr>
        <th colspan="2">CoVoST2</th>
        <th colspan="2">EuroParlST</th>
        <th colspan="4">CoVoST2</th>
        <th colspan="4">EuroParlST</th>
    </tr>
    <tr>
        <th> </th>
        <th> </th>
        <th> </th>
        <th colspan="2">en-de</th>
        <th colspan="2">en-zh</th>
        <th colspan="2">en-de</th>
        <th colspan="3">en-it</th>
    </tr>
    <tr>
        <th>WER</th>
        <th>CER</th>
        <th>WER</th>
        <th>CER</th>
        <th>BLEU</th>
        <th>COMET</th>
        <th>BLEU</th>
        <th>COMET</th>
        <th>BLEU</th>
        <th>COMET</th>
        <th>BLEU</th>
        <th>COMET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>A.1 (ASR/ST)</td>
        <td>7.84</td>
        <td>3.70</td>
        <td>11.21</td>
        <td>7.30</td>
        <td>30.24</td>
        <td>0.789</td>
        <td>40.14</td>
        <td>0.806</td>
        <td>25.06</td>
        <td>0.840</td>
        <td>27.55</td>
        <td>0.860</td>
    </tr>
    <tr>
        <td>A.2 (ASR/ST/SQA)</td>
        <td>7.21</td>
        <td>3.44</td>
        <td>10.98</td>
        <td>7.18</td>
        <td>30.68</td>
        <td>0.789</td>
        <td>40.63</td>
        <td>0.807</td>
        <td>25.21</td>
        <td>0.841</td>
        <td>28.36</td>
        <td>0.859</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>7.59</td>
        <td>3.55</td>
        <td>11.20</td>
        <td>7.20</td>
        <td>31.42</td>
        <td>0.770</td>
        <td>42.19</td>
        <td>0.802</td>
        <td>27.16</td>
        <td>0.809</td>
        <td>27.86</td>
        <td>0.842</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>7.01</td>
        <td>3.25</td>
        <td>10.82</td>
        <td>7.11</td>
        <td>31.83</td>
        <td>0.772</td>
        <td>42.36</td>
        <td>0.804</td>
        <td>26.70</td>
        <td>0.812</td>
        <td>27.77</td>
        <td>0.848</td>
    </tr>
  </tbody>
</table>

Table 6: ASR and ST results for the test sets of CoVoST2 and EuroParlST.


<table>
  <thead>
    <tr>
        <th rowspan="2">Model (fine-tuning data)</th>
        <th rowspan="2">ASR (WER)<br/>en</th>
        <th colspan="3">ST/MT (BLEU)</th>
        <th colspan="3">ST/MT (COMET)</th>
    </tr>
    <tr>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
    </tr>
    <tr>
        <th colspan="8">Text-only models (MT/QA)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>Llama-3.1-8B-Instruct (zero-shot)</td>
        <td>-</td>
        <td>21.27</td>
        <td>33.81</td>
        <td>44.01</td>
        <td>0.732</td>
        <td>0.757</td>
        <td>0.755</td>
    </tr>
    <tr>
        <td>B Text-only LoRA (MT/QA)</td>
        <td>-</td>
        <td>36.94</td>
        <td>49.81</td>
        <td>52.33</td>
        <td>0.782</td>
        <td>0.815</td>
        <td>0.822</td>
    </tr>
    <tr>
        <th colspan="8">Speech-only models (ASR/ST/SQA)</th>
    </tr>
    <tr>
        <td>SeamlessM4T-v2-large</td>
        <td>25.3</td>
        <td>23.77</td>
        <td>37.84</td>
        <td>28.17</td>
        <td>0.669</td>
        <td>0.713</td>
        <td>0.663</td>
    </tr>
    <tr>
        <td>A.1 Speech Projector (ASR/ST)</td>
        <td>15.9</td>
        <td>26.77</td>
        <td>36.34</td>
        <td>38.65</td>
        <td>0.718</td>
        <td>0.750</td>
        <td>0.753</td>
    </tr>
    <tr>
        <td>A.2 Speech Projector (ASR/ST/SQA)</td>
        <td>15.9</td>
        <td>26.85</td>
        <td>36.09</td>
        <td>38.85</td>
        <td>0.720</td>
        <td>0.749</td>
        <td>0.753</td>
    </tr>
    <tr>
        <th colspan="8">Multimodal models (ASR/ST/SQA)</th>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/MT/SQA/QA)</td>
        <td>**13.9**</td>
        <td>28.74</td>
        <td>41.73</td>
        <td>40.72</td>
        <td>0.716</td>
        <td>**0.759**</td>
        <td>0.756</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>14.4</td>
        <td>27.20</td>
        <td>**42.01**</td>
        <td>41.23</td>
        <td>0.712</td>
        <td>0.752</td>
        <td>0.753</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/SQA/QA)</td>
        <td>17.2</td>
        <td>**29.18**</td>
        <td>39.14</td>
        <td>40.99</td>
        <td>0.719</td>
        <td>0.755</td>
        <td>0.762</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>15.5</td>
        <td>27.62</td>
        <td>33.22</td>
        <td>**41.74**</td>
        <td>**0.726**</td>
        <td>0.749</td>
        <td>**0.764**</td>
    </tr>
  </tbody>
</table>

Table 7: ACL 60-60 dev set results for the different models and backbones used in this work.


<table>
  <thead>
    <tr>
        <th rowspan="2"> </th>
        <th colspan="4">Valid Questions</th>
        <th colspan="4">Invalid Questions</th>
    </tr>
    <tr>
        <th>en-en</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
        <th>en-en</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>A.1 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>0.975</td>
        <td>0.532</td>
        <td>0.541</td>
        <td>0.665</td>
        <td>0.999</td>
        <td>0.984</td>
        <td>0.990</td>
        <td>0.988</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>0.975</td>
        <td>0.536</td>
        <td>0.546</td>
        <td>0.666</td>
        <td>0.999</td>
        <td>0.990</td>
        <td>0.989</td>
        <td>0.991</td>
    </tr>
  </tbody>
</table>

Table 8: BERT scores for SpokenSQuAD test sets computed using the same settings from the organizers (default model, bert_score version 0.3.13). Invalid questions corresponds to a version of the test set in which the questions are impossible to answer given the speech context.


<table>
  <thead>
    <tr>
        <th rowspan="2"> </th>
        <th colspan="4">Valid Questions</th>
        <th colspan="4">Invalid Questions</th>
    </tr>
    <tr>
        <th>en-en</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
        <th>en-en</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>A.1 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>0.857</td>
        <td>0.869</td>
        <td>0.863</td>
        <td>0.899</td>
        <td>0.996</td>
        <td>0.997</td>
        <td>0.996</td>
        <td>0.997</td>
    </tr>
    <tr>
        <td>A.2 + B (ASR/ST/MT/fluentSQA/fluentQA)</td>
        <td>0.857</td>
        <td>0.869</td>
        <td>0.863</td>
        <td>0.896</td>
        <td>0.995</td>
        <td>0.998</td>
        <td>0.998</td>
        <td>0.998</td>
    </tr>
  </tbody>
</table>

Table 9: BERT scores for SpokenSQuAD test sets computed using xlm-roberta-large. All other settings are equal to Table 8. Invalid questions corresponds to a version of the test set in which the questions are impossible to answer given the speech context.

12

NAVER LABS Europe Submission to the Instruction-following Track

# C. Models Hyperparameters

Table 12 lists the data splits used for each model presented in results Table 2. Table 13 presents the probability sampling employed during training.

## C.1. Speech Projector (A) Hyperparameters

**Architecture** We explored multiple architectures to map speech embeddings from SeamlessM4T-v2-large to Llama-3.1-8B-Instruct. To train the speech projector, speech features extracted from SeamlessM4T-v2-large are input to the projector, and the resulting outputs are passed through a frozen Llama-3.1-8B-Instruct model. The speech projector, initialized with random weights, is trained using cross-entropy loss on an ASR task. Preliminary experiments demonstrated that the Transformer encoder architecture consistently outperformed both the Conformer and Multi-Layer Perceptron architectures of similar parameter sizes. Consequently, we adopt the Transformer encoder architecture for all experiments presented in this work.

**Averaged Features** As mentioned, in preliminary experiments, we experimented using the original output of SeamlessM4T-v2-large, as well as performing average every 2 or 3 frames. We observe that averaging every 3 frames results in models that are considerably faster to train, while maintaining similar performance to the original output.

**Data Ratio** For the ASR task, preliminary experiments revealed that training solely on EuroparlST ASR data resulted in poor generalization, whereas incorporating CoVoST ASR data significantly improved model robustness. For the ST task, we defined the data sampling ratios according to the target language distribution across the CoVoST and EuroparlST datasets.

**Batch Size** ASR and ST tasks use a batch size of 16, while SQA is batched with size 8, due to the longer user prompts.

**Checkpoint Selection** Checkpoints were selected based on development set performance across three or four configurations: ASR-best, ST-best, SQA-best (A.2-only), and an All-best checkpoint combining all tasks. We only present results for ST-best checkpoints, which we found to produce the best scores in ST compared to the other versions, while only marginally decreasing scores in ASR compared to the ASR-best checkpoint. We do not consider SQA-best checkpoints, as the overall

SQA performance of projector-only models is very low regardless of the checkpoint selection method.

**Exclusion of Chinese SQA Data** During the training of the speech projector (A.2), we excluded Chinese SQA data. This was due to parallel observation in B models (text-only), in which we observed that the LLM failed to generate coherent Chinese answers. While later we were able to confirm that the issue did not come from the Chinese split itself, this model was obtained simultaneously to that finding, explaining why the data was not included in this setting.

## C.2. Multimodal Models (C) Hyperparameters

In this section we present some ablation experiments for our multimodal adaptation setup. The experiments are performed by producing variants of the A.1+B model, which is the model we submitted to the challenge. Table 10 present ACL 60-60 dev split ASR and ST results that are discussed in the next paragraphs.

**Impact of Parameters Count** During our multimodal merging step, we combine text-only LoRA weights with our speech projector, yielding better scores. Since this increase in scores could be simply due the additional parameters, we trained a variant of our model in which the merging step is performed using a randomly initialized LoRA. We observe that our training setup indeed benefits from any additional weights during adaptation: the models trained with a randomly initialized LoRA outperform the speech projector backbone (A.1). Adding textual tasks in this setting does not help the system, which we hypothesize is due to the LoRA weights not being pretrained on the textual task. Finally, adapting using a pretrained LoRA model further improves ASR and ST scores for two out of three language directions (en-it and en-zh).

**Impact of Textual Tasks** For the multimodal models presented in the main paper, we adapt pretrained modules leveraging speech and textual tasks. We thus investigated the impact of having aligned speech and textual tasks during this adaptation. We observe that incorporating textual tasks has little impact on ASR performance, while substantially improving ST performance for Italian. The results are less favorable for German and Chinese: in German, the addition of textual tasks leads to a performance drop, whereas in Chinese, the decline is minimal. Overall, these findings suggest that the textual modality may be particularly beneficial in low-resource settings. Italian, which has the fewest training examples in our dataset, appears to benefit the most from this adaptation.

13

NAVER LABS Europe Submission to the Instruction-following Track

<table>
  <thead>
    <tr>
        <th> </th>
        <th colspan="2">ASR</th>
        <th colspan="3">ST (BLEU)</th>
        <th colspan="3">ST (COMET)</th>
    </tr>
    <tr>
        <th> </th>
        <th>WER</th>
        <th>CER</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
        <th>en-de</th>
        <th>en-it</th>
        <th>en-zh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>A.1 (ASR/ST)</td>
        <td>15.9</td>
        <td>7.7</td>
        <td>26.77</td>
        <td>36.34</td>
        <td>38.65</td>
        <td>0.718</td>
        <td>0.750</td>
        <td>0.753</td>
    </tr>
    <tr>
        <td>A.1 + random LoRA (ASR/ST/SQA)</td>
        <td>17.4</td>
        <td>8.2</td>
        <td>29.07</td>
        <td>37.04</td>
        <td>41.47</td>
        <td>0.732</td>
        <td>0.760</td>
        <td>0.761</td>
    </tr>
    <tr>
        <td>A.1 + random LoRA (ASR/ST/SQA) + (MT/QA)</td>
        <td>16.4</td>
        <td>7.7</td>
        <td>28.07</td>
        <td>38.88</td>
        <td>41.24</td>
        <td>0.725</td>
        <td>0.755</td>
        <td>0.760</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST)</td>
        <td>13.8</td>
        <td>6.4</td>
        <td>27.91</td>
        <td>28.88</td>
        <td>41.70</td>
        <td>0.728</td>
        <td>0.743</td>
        <td>0.760</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST) + (MT)</td>
        <td>14.9</td>
        <td>7.2</td>
        <td>26.09</td>
        <td>32.97</td>
        <td>41.10</td>
        <td>0.715</td>
        <td>0.741</td>
        <td>0.755</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/SQA)</td>
        <td>14.1</td>
        <td>6.5</td>
        <td>29.02</td>
        <td>30.08</td>
        <td>41.85</td>
        <td>0.722</td>
        <td>0.743</td>
        <td>0.763</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/SQA) + (MT/QA)</td>
        <td>14.4</td>
        <td>6.5</td>
        <td>27.20</td>
        <td>42.01</td>
        <td>41.23</td>
        <td>0.712</td>
        <td>0.752</td>
        <td>0.753</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/SQA) + (MT/QA) No synthetic data</td>
        <td>18.8</td>
        <td>9.2</td>
        <td>28.47</td>
        <td>32.97</td>
        <td>40.03</td>
        <td>0.717</td>
        <td>0.741</td>
        <td>0.753</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/SQA) + (MT/QA) Only synthetic data</td>
        <td>13.9</td>
        <td>6.6</td>
        <td>29.71</td>
        <td>39.80</td>
        <td>42.04</td>
        <td>0.721</td>
        <td>0.751</td>
        <td>0.754</td>
    </tr>
    <tr>
        <td>A.1 + B (ASR/ST/SQA) + (MT/QA) 2K steps</td>
        <td>14.3</td>
        <td>6.4</td>
        <td>27.09</td>
        <td>40.53</td>
        <td>41.53</td>
        <td>0.713</td>
        <td>0.753</td>
        <td>0.755</td>
    </tr>
  </tbody>
</table>

Table 10: ACL 60-60 dev set ASR and ST scores for variants of our best model (A.1+B).


<table>
  <thead>
    <tr>
        <th>ASR/ST/SQA</th>
        <th>average WER</th>
        <th>average BLEU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>0.2 / 0.4 / 0.4</td>
        <td>16.03</td>
        <td>37.58</td>
    </tr>
    <tr>
        <td>0.2 / 0.5 / 0.3</td>
        <td>17.72</td>
        <td>36.36</td>
    </tr>
    <tr>
        <td>0.2 / 0.6 / 0.2</td>
        <td>16.24</td>
        <td>36.62</td>
    </tr>
    <tr>
        <td>0.25 / 0.5 / 0.25</td>
        <td>16.58</td>
        <td>37.08</td>
    </tr>
    <tr>
        <td>0.3 / 0.5 / 0.2</td>
        <td>16.23</td>
        <td>37.30</td>
    </tr>
  </tbody>
</table>

Table 11: ACL 60-60 dev and test set average WER and BLEU scores for our best model (A.1+B) by varying the ASR/ST/SQA ratios.

data sourced from EuroParlST and CoVoST2.

**Impact of SQA** Examining the results in Table 10, we observe that incorporating the SQA and QA tasks leads to improvements in both ASR and ST performance. We hypothesize that the SQA task enhances the model's adherence to the prompt by encouraging it to *attend to* the information provided, thereby reducing both task and language confusion.

**Number of Adaptation Steps** Table 10 presents results for a version of our model trained for twice as long (2K steps). At this point, we observe signs of training saturation: differences in ASR and COMET scores across all metrics are minimal, and BLEU scores drop for both German and Italian. These results suggest that our adaptation step does not require a considerable number of training steps.

**Task Ratios** On our preliminary experiments, we tested different task ratios, selecting the one with best average WER and BLEU scores over both ACL 60-60 dev and test set. Table 11 presents those results.

**Inclusion of Synthetic Textual Data** Table 11 presents the results of our investigation into the inclusion of potentially noisy synthetic textual data during training. We observe that excluding this synthetic data (*No synthetic data*) negatively affects both BLEU and WER scores. Conversely, training exclusively with synthetic data (*Only synthetic data*) yields improved performance across all metrics. We attribute this to the fact that the target text in the non-synthetic data is duplicated from the speech task (i.e. the MT set is built from the ST set), leading to reduced data diversity. Removing this duplicated data introduces greater variability during the adaptation phase, which appears beneficial. Finally, using both types of data leads to improved performance for all languages except German. As discussed in the main paper, we hypothesize that this discrepancy is due to issues in the German training

### C.2.1 Post-submission Discussion

Due to time constraints, many of our ablation studies were conducted after the initial submission. Upon analyzing these results (Table 10), we hypothesize that there may be an issue with the German training data: the more the model is exposed to it during training, the worse the COMET scores become. This hypothesis is supported by our *Only synthetic data* results, which show improved BLEU scores for German when we exclude textual data from EuroParlST and CoVoST2. Additionally, our ablations suggest that using textual data selectively, rather than uniformly, may be more effective. In particular, textual supervision appears to be most beneficial for Italian, with more limited gains observed for the other two language directions.

Regarding SQA, we were surprised to find that the evaluation setup provided by the organizers yields scores that differ significantly from those obtained with our own evaluation protocol (see Tables 8 and 9). These discrepancies also extend to the scores we obtain using

14

NAVER LABS Europe Submission to the Instruction-following Track

<table>
  <thead>
    <tr>
        <th></th>
        <th colspan="3">CoVoST</th>
        <th colspan="3">EuroParlST</th>
        <th colspan="4">SpokenSQuAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>Model</td>
        <td>ASR</td>
        <td>ST</td>
        <td>MT</td>
        <td>ASR</td>
        <td>ST</td>
        <td>MT</td>
        <td>ASR</td>
        <td>MT</td>
        <td>SQA/QA</td>
        <td>*fluent* SQA/QA</td>
    </tr>
    <tr>
        <td>B Text-only LoRA</td>
        <td>✗</td>
        <td>✗</td>
        <td>✓</td>
        <td>✗</td>
        <td>✗</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
    </tr>
    <tr>
        <td>A.1 Speech Projector (ASR/ST)</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
        <td>✗</td>
        <td>✗</td>
        <td>✗</td>
    </tr>
    <tr>
        <td>A.2 Speech Projector (ASR/ST/SQA)</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓ (no zh)</td>
        <td>✗</td>
    </tr>
    <tr>
        <td>A.1 + B Multimodal model (ASR/ST/MT/SQA/QA)</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
    </tr>
    <tr>
        <td>A.1 + B Multimodal model (ASR/ST/MT/*fluent*SQA/*fluent*QA)</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>A.2 + B Multimodal model (ASR/ST/MT/SQA/QA)</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
    </tr>
    <tr>
        <td>A.2 + B Multimodal model (ASR/ST/MT/*fluent*SQA/*fluent*QA)</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✗</td>
        <td>✓</td>
    </tr>
  </tbody>
</table>
Table 12: List of datasets and splits used for each model presented in Table 2. Statistics for number of examples can be seen in Table 1.

<table>
  <thead>
    <tr>
        <th rowspan="3"></th>
        <th>ASR</th>
        <th colspan="4">ST/MT</th>
        <th colspan="5">SQA/QA (valid/invalid)</th>
    </tr>
    <tr>
        <th>task</th>
        <th>task</th>
        <th rowspan="2">en-de</th>
        <th rowspan="2">en-it</th>
        <th rowspan="2">en-zh</th>
        <th>task</th>
        <th rowspan="2">en-en</th>
        <th rowspan="2">en-de</th>
        <th rowspan="2">en-it</th>
        <th rowspan="2">en-zh</th>
    </tr>
    <tr>
        <th>ratio</th>
        <th>ratio</th>
        <th>ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>Text-only LoRA</td>
        <td>x</td>
        <td>0.6</td>
        <td>0.4</td>
        <td>0.3</td>
        <td>0.3</td>
        <td>0.4</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
    </tr>
    <tr>
        <td>A.1</td>
        <td>0.4</td>
        <td>0.6</td>
        <td>0.3</td>
        <td>0.4</td>
        <td>0.3</td>
        <td>x</td>
        <td>x</td>
        <td>x</td>
        <td>x</td>
        <td>x</td>
    </tr>
    <tr>
        <td>A.2</td>
        <td>0.4</td>
        <td>0.35</td>
        <td>0.3</td>
        <td>0.4</td>
        <td>0.3</td>
        <td>0.25</td>
        <td>0.2 / 0.2</td>
        <td>0.15 / 0.15</td>
        <td>x</td>
        <td>0.15 / 0.15</td>
    </tr>
    <tr>
        <td>A.1 + B</td>
        <td>0.2</td>
        <td>0.4</td>
        <td>0.4</td>
        <td>0.3</td>
        <td>0.3</td>
        <td>0.4</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
    </tr>
    <tr>
        <td>A.2 + B</td>
        <td>0.2</td>
        <td>0.4</td>
        <td>0.4</td>
        <td>0.3</td>
        <td>0.3</td>
        <td>0.4</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
        <td>0.2 / 0.05</td>
    </tr>
  </tbody>
</table>
Table 13: Two-level sampling ratio for each model.

LLM-as-judge (Table 2). We plan to further investigate the limitations of our current evaluation setup to better understand these inconsistencies.

15