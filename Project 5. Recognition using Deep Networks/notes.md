# System Center: Unified Infrastructure Resource Management

## Problem Statement

Operators today maintain 30k infrastructure resource nodes. We need a unified platform to centralize fleet management, giving better control across all resources to help us achieve better operational efficiency. The platform should also support diagnostics and reporting, either on a schedule or on-demand. This shift enables more reliable, proactive infrastructure management.

## Background

This need brings out our project, **System Center**. Our team, **CIE**, aims to improve how we observe and manage a global fleet of data centers and labs that support our GPUs, CPUs, and Tegra systems. Our goal is to simplify and unify how resources are governed across these environments, making management more consistent and efficient. 

We’ve built a centralized platform that collects resource data from across the infrastructure, providing insights with an emphasis on AI-driven solutions. Ultimately, it’s all about enhancing operational efficiency and ensuring we make the most of our compute resources.

## Motivation

I worked on the **Insight Engine**, specifically on generating high-level summaries of node status—particularly for bare metal nodes—and identifying trends. As shown in the workflow, we collect metadata from bare metal nodes, store it in a centralized data lake, and then ingest it into the Insight Engine. The engine produces AI-enhanced insights, which are then consumed by systems like **Colossus** and **VAL** to help the teams quickly identify and diagnose issues.

## Methodology

### Options:

- **Traditional ML Models**: Require structured data, feature engineering, and model training for each specific task.
- **LLM-Based Services**: Can adapt to multiple tasks with minimal tuning, reducing development and maintenance overhead.

### Solution Architecture

We collect data and attributes from **Colossus**, our metadata source, and input them into structured prompts. Using **prompt orchestrators**, we manage the execution of these prompts through the **LLM service** to generate insights for each resource node. Additionally, we incorporate external evaluators to provide confidence scores for the generated node summaries. This allows us to refine the prompts and compare models to select the most accurate and meaningful results.

## Implementations

### Step-by-Step:

1. **Data Ingestion**: We pull static metadata from **Colossus**, collecting GPU utilization and usage attributes using the **NVIDIA-SMI** tool, and use **Cursor** to generate synthetic dynamic time-series data.
2. **Insight Generation**: We designed a structured schema for node summaries, health, performance, recommendations, references, and executive summaries, using models like **NIM** and **LLMGateway**.
3. **Result Evaluation**: We investigate a reasoning model and the **DeepEval** framework to score the outputs, helping us measure confidence and refine our prompts for better accuracy.

## Data Collection

We use the **Colossus CLI** as our main tool for querying hardware leasing inventory. It pulls in a focused set of data—specifically 24 key attributes—giving us a clear picture of the hardware's status and usage. These attributes cover areas like resource, GPU/CPU metrics, system software, and overall utilization.

## Models

We feed the data into different models depending on the use case:

- **NIM** powered by **LLaMA 3.1**: Offers full control over performance tuning and fast query responses. On a test set of 582 entries, the average query time was just 1 second.
- **LLMGateway**: A unified API to access hosted models, with an average query time of 12 seconds, suitable for complex logical reasoning and multi-step thinking tasks.

### Model Strengths

- **Internal Models**: Ideal for fast, factual summarization.
- **Hosted Models**: Shine in complex reasoning tasks.

Thus, we use **NIM** to generate node summaries, as it’s optimized for delivering precise, factual insights efficiently.

## Prompt Techniques Evaluation

### Techniques Used:
- **Few-shot Prompting**: For Health and Performance summaries, to provide the model with examples of expected formats.
- **Chain-of-Thought**: For generating Recommendations, guiding the model step by step through problem identification, analysis, and suggestions.
- **Prompt Chaining**: For generating the Executive Summary, combining the generated sections into a cohesive report.

## Evaluation

Once the node summary is generated, we evaluate the quality of the output using the following approaches:

1. **Reasoning Model (e.g., GPT-4o)**: Assesses faithfulness, relevance, and accuracy of the summary.
2. **DeepEval**: A RAG-based, embedding method for natural language reasoning and quality assessment.

Since we are not currently using a RAG-based solution, we primarily rely on the reasoning model for evaluation.

## Results/Demo

Here is the sample node summarization result for our **System Center MVP**.

**Demo:** [Link to Demo]

## Impacts

The goal is to provide clear visibility and actionable intelligence to operators. By collecting and normalizing key attributes, we ensure consistent data for analysis. Structured prompts generate relevant insights tailored to each node, and evaluation metrics help select the most accurate, high-impact summaries from multiple models.

## Learnings and Findings

- We worked with tools like **NIM**, **LLM**, **DeepEval**, and the **NVAI framework**.
- Automated data ingestion from **Colossus** streamlined the pipeline.
- We designed structured prompts, compared different models, and ran performance load tests to ensure system reliability.

## What's Next?

- Expanding data capture to include a wider range of dynamic, machine-level signals for deeper analysis.
- Enhancing the knowledge base to generate more comprehensive and actionable insights, with a built-in feedback mechanism for continuous improvement.

## Acknowledgements

I’d like to give credit to my mentor **Prathik**, **Stanley**, my manager **Gobi**, and our **CIE team**—**Krishnan** and **Shivam**—for their support, guidance, and encouragement throughout the internship.

## Q&A

### 1. Faithfulness, Accuracy, and Relevance

- **Relevance**: Ensures the output focuses on the most important aspects and remains on-topic.
    - *Example*: A summary of GPU performance should mention GPU utilization, temperature, and usage patterns, avoiding unrelated topics.
  
- **Accuracy**: Focuses on the correctness of the details in the generated content.
    - *Example*: A summary that reports "CPU usage at 50%" is accurate if it's within a reasonable range for the system.
  
- **Faithfulness**: Ensures the generated content remains true to the underlying data and doesn’t introduce false or misleading information.
    - *Example*: If a summary says "The system has been running for 30 days without issues," but logs show downtime on day 15, that would be unfaithful.

---

Thank you for your time!
