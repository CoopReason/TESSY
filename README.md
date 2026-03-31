<div align="center">

<table>
  <tr>
    <td align="center" width="140">
      <img src="figs/logo.png" alt="TESSY Logo" width="110"/>
    </td>
    <td align="left">
      <h1 style="margin-bottom: 6px;">
        TESSY
      </h1>
      <h3 style="margin-top: 0; font-weight: normal;">
        Teacher–Student Cooperative Synthesis Framework
      </h3>
      <p style="margin-top: 6px;">
        <em>On-policy Data Synthesis for Reasoning Models</em>
      </p>
    </td>
  </tr>
</table>

<br/>

📄 <b>Paper</b>:  
<a href="https://github.com/CoopReason/TESSY/blob/main/paper/TESSY.pdf">
How to Fine-Tune a Reasoning Model?<br/>
A Teacher–Student Cooperation Framework to Synthesize Student-Consistent SFT Data
</a>

<br/>

🤗 <b>Dataset</b>: 
<a href="https://huggingface.co/datasets/CoopReason/TESSY-Code-80K">
TESSY-Code-80K
</a>

</div>
---

## 🚀 Overview

Training reasoning models (e.g., Qwen3) is highly sensitive to the **data distribution**.  
We observe that:

> ❗ Using off-policy data (e.g., directly from a strong teacher model) for SFT can lead to **severe catastrophic forgetting**, especially for code generation tasks.

---

## 💡 Key Idea

To address this issue, we propose **TESSY**, a **Teacher–Student Cooperative Synthesis framework** that generates *on-policy* training data.

Instead of letting a teacher fully generate training samples, TESSY **decouples generation into two parts**:

- 🧠 **Teacher model** → generates *reasoning trajectories*
- ✍️ **Student model** → generates *non-reasoning content*

This ensures:
- Alignment with student distribution (on-policy)
- Preservation of teacher reasoning quality

---

## 🧩 Method

<div align="center">
  <img src="figs/overview.png" alt="TESSY Overview" width="720"/>
</div>

TESSY performs **iterative cooperative generation**:

1. Predict reasoning boundaries  
2. Alternate between:
   - Teacher → reasoning steps  
   - Student → final outputs / transitions  
3. Construct full reasoning trajectories aligned with the student model  

---

## 📊 Results

<div align="center">
  <img src="figs/main_results.png" alt="Main Results" width="560"/>
</div>

- Direct SFT using **GPT-OSS-120B data** (Teacher-Only) → ❌ catastrophic forgetting  
- TESSY-synthesized data → ✅ significant improvement on code generation  

---

## 📦 Released Dataset

We release the dataset used in our paper:

- **Name:** TESSY-Code-80K  
- **Designed for:** Qwen3-8B  
- **Effect:** Significant improvement on code tasks  

🔗 https://huggingface.co/datasets/CoopReason/TESSY-Code-80K

> Note: While applicable to other Qwen3 models, gains may vary since synthesis is tailored to Qwen3-8B.

---

## ⚙️ Setup & Usage

### 1. Start Model Servers

Record the API endpoints (IP + port) of both servers.

Adjust the following parameters based on your hardware setup:
- `TP`
- `GPU_MEM_UTILIZATION`

---

### 2. Prepare Boundary Predictors

We provide pretrained predictors:

- `CoopReason/Boundary_Predictor_Teacher_Code`
- `CoopReason/Boundary_Predictor_Student_Code`

You can also train your own predictors using:

```bash
Boundary_Predictor/
```
3. Run TESSY
```
bash run_tessy.sh \
  datas/examples.jsonl \
  results/example_outputs.jsonl \
  http://127.0.0.1:23333/v1/completions \
  http://127.0.0.1:23334/v1/completions
```

Example input: datas/examples.jsonl (subset of OJBench)
Output: results/example_outputs.jsonl


⚠️ Notes
This repository is currently a research prototype (demo version)
Built on top of vLLM
Further improvements possible in:
inference efficiency
scheduling
batching strategies
🤝 Contact & Future Work

We are actively improving TESSY and welcome:

Feedback
Collaboration
Real-world deployment discussions

Feel free to reach out!

📌 Citation

Coming Soon

