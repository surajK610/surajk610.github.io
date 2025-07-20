---
layout: page
title: hacking-gpt
description: using mechinterp to hack a PPO-ed GPT-2
img: assets/img/ppoed_models/ppo_actor_crit.png
importance: 6
category: spring 2024
---

# Are PPO-ed Language Models Hackable?

*Exploring how reinforcement learning "aligns" language models and whether this alignment can be circumvented through mechanistic understanding. This research was written in spring of 2024 (back when RLHF was still somewhat popular).*

## The Alignment Challenge

Large language models come with a fundamental problem: they often exhibit undesirable behaviors like toxicity, bias, and negative sentiment. This has led to the widespread adoption of **Reinforcement Learning from Human Feedback (RLHF)** and specifically **Proximal Policy Optimization (PPO)** to "align" these models with human preferences.

The critical question we are interested in: *Are these alignment techniques actually removing harmful capabilities, or are they simply learning to hide them?*

Our research tackles this question head-on by examining GPT-2 through the lens of **mechanistic interpretability** before and after PPO training. What we discovered has important implications for AI safety and the robustness of current alignment methods. Recent research by [Lee et al. (2024)](https://arxiv.org/abs/2401.01967) discovered that models aligned with Direct Preference Optimization (DPO) to avoid toxicity don't actually remove toxic knowledge—they just learn an "offset" to avoid expressing it. We extend this finding to examine whether PPO exhibits similar behavior with sentiment generation.

## A Controlled Study of Sentiment Alignment

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ppoed_models/ppo_actor_crit.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Here is the actor-critic model that has been used for PPO. Specifically, we use
</div>

Rather than studying the complex domain of toxicity directly, we designed a controlled experiment focusing on **sentiment generation**:

- **Base Model**: GPT-2 fine-tuned on the IMDB movie review dataset
- **Alignment Target**: Generate positive sentiment responses instead of negative ones
- **Reward Model**: A DistilBERT classifier trained to detect positive sentiment
- **Experimentation**: Full access to model weights and activations (white-box analysis)

This setup allowed us to peer inside the model's "mind" and understand exactly how PPO changes the underlying computations.

## Mechanistic Analysis

We used mechanistic interpreptability techniques to conduct our analysis.

### Step 1: Identifying "Negative" Weights

Using insights from recent mechanistic interpretability research, we identified which parts of GPT-2 are responsible for generating negative sentiment. Here's how:

1. **Trained a sentiment probe** on the model's internal representations
2. **Found "negative" value vectors** using cosine similarity with the probe weights
3. **Mapped these vectors to vocabulary space** to see what concepts they represent

```python
def find_negative_weight_meanings(tokenizer, w_negative, hooked_model, k=5, n_weights=10):
    _, scores_gpt2 = get_svd(hooked_model, w_negative, 128)
    
    vectors_of_interest = [
        (_score_obj[2], _score_obj[1], _score_obj[0])
        for _score_obj in scores_gpt2[:64]
    ]
    
    topn_negative_value_weights = vectors_of_interest[:n_weights]
    
    for layer, idx, _ in topn_negative_value_weights:
       decode_topk(tokenizer, hooked_model, layer, idx, k)
```

**What we found**: Specific neurons in layers 6-9 consistently activated for negative concepts like "useless," "disastrous," "fail," and "bad."

| Layer | Index | Top Negative Tokens |
|-------|-------|-------------------|
| 7 | 2394 | useless, mediocre, worthless |
| 6 | 2360 | unus, disastrous, deteriorated |
| 9 | 3047 | negligible, diminished, fewer |
| 7 | 2464 | fail, Fail, Wrong |



### Step 2: PPO Training Results

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ppoed_models/sentiment_change.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Histogram of sentiment of responses in heldout test set pre and post PPO the GPT-2.
</div>


After PPO training, our model successfully learned to generate positive sentiment:
- **Original model**: Average sentiment score of 0.27
- **PPO-trained model**: Average sentiment score of 0.80

The alignment worked! But *how* did it work?

## PPO Doesn't Remove Negative Weights

Here's where our investigation revealed something surprising about how PPO actually works:

### Finding #1: Weights Are Barely Changed

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ppoed_models/weight_change.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Weights are minimally changed by Proximal Policy Optimization of the full GPT-2 model.
</div>

```python
# Calculate cosine similarity between original and PPO weights
sims = []
for name, param in model.state_dict().items():
    if 'h.11.' in name or 'h.10.' in name:
        param_ppo = trainer.model.state_dict()['base_model.' + name]
        curr_sim = F.cosine_similarity(param, param_ppo, dim=0).cpu()
        sims.extend(list(curr_sim))
```

**Result**: Almost all weights maintained cosine similarity ≥ 0.9998 with their original values!

### Finding #2: Negative Weights Remain Intact

The "negative" weights we identified earlier? They were virtually unchanged after PPO training:

| Layer | Index | Original Tokens | PPO-ed Tokens |
|-------|-------|----------------|---------------|
| 7 | 2394 | useless, mediocre, worthless | useless, mediocre, worthless |
| 6 | 2360 | unus, disastrous, deteriorated | unus, disastrous, deteriorated |
| 9 | 3047 | negligible, diminished, fewer | negligible, diminished, fewer |

**Implication**: PPO didn't remove the model's knowledge of negative concepts—it just learned to avoid expressing them.

### Finding #3: Activations Change, Not Weights

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/ppoed_models/shit_prediction.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/ppoed_models/activation_change.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (Left) Logit lens shows that negative concept activations scaled down across model residual stream (this is specifically for the word sh*t). (Right) The activation differences for the ten most ‘negative’ value vectors.
</div>


Using the "logit lens" technique, we tracked how concepts flow through the model's residual stream:

```python
# Track negative concept activations across layers
def logit_lens_analysis(model, negative_vectors, prompts):
    activations = {}
    for layer in range(model.cfg.n_layers):
        # Project intermediate representations to vocabulary space
        logits = model.unembed(residual_stream[layer])
        activations[layer] = track_negative_concepts(logits, negative_vectors)
    return activations
```

**Discovery**: Negative concepts were still computed internally but were systematically suppressed in the residual stream as information flowed through the model.

## The Hack: Exploiting PPO's Weakness

Armed with this mechanistic understanding, we designed a "jailbreak" to force the aligned model to generate negative sentiment:

### Activation Scaling Attack

```python
def activation_scaling_hack(model, negative_vectors, scale_factor=10):
    """
    Scale up activations corresponding to negative value vectors
    """
    def hook_fn(module, input, output):
        # Identify negative concept activations
        for layer, idx in negative_vectors:
            if is_negative_activation(output, layer, idx):
                # Scale up by factor of 10
                output[:, :, idx] *= scale_factor
        return output
    
    # Register hooks on MLP layers
    for layer in range(6, 10):  # Layers containing negative weights
        model.blocks[layer].mlp.register_forward_hook(hook_fn)
```

**Result**: By scaling the activations of negative value vectors by a factor of 10, we successfully "jailbroke" the aligned model:
- **PPO-aligned model**: 0.80 average sentiment score
- **After activation scaling**: 0.43 average sentiment score (back to negative!)

## Implications: What This Means for AI Safety

Our findings reveal a fundamental limitation in current alignment approaches:

### 1. **Alignment ≠ Capability Removal**
PPO doesn't actually remove harmful capabilities—it learns a "wrapper" that avoids expressing them. The underlying knowledge remains intact and potentially exploitable.

### 2. **White-Box Vulnerability**
If an adversary has access to model weights (increasingly common with open-source models), they can potentially reverse-engineer alignment measures.

### 3. **The Offset Problem**
Similar to findings in other research, PPO learns an "offset" that shifts the model's behavior without fundamentally altering its internal representations.

## Attempted Solution: Regularization-Based Mitigation

We experimented with modifying the PPO objective to actively penalize negative weights:

```python
# Modified reward function
r_modified = r_original - λ₁ * r_KL + λ₂ * Σ(||w - w_original||)
#                                    negative weights
```

Where:
- `r_original`: Standard PPO reward (positive sentiment)
- `λ₁ * r_KL`: KL divergence penalty (standard)
- `λ₂ * Σ(||w - w_original||)`: Penalty for preserving negative weights

**Challenge**: Finding the right balance proved difficult:
- **λ₂ too low**: Negative weights remain unchanged
- **λ₂ too high**: Model loses coherent language generation capabilities

## Future Directions: Toward More Robust Alignment

Our research suggests several promising directions:

### 1. **Mechanistic-Informed Alignment**
- Use interpretability tools to identify and specifically target harmful representations
- Design training objectives that explicitly modify problematic weights

### 2. **Adversarial Robustness Testing**
- Systematically test aligned models against mechanistic attacks
- Develop benchmarks for alignment robustness beyond behavioral evaluation

### 3. **Fundamental Architecture Changes**
- Explore architectures that make harmful capabilities truly removable
- Investigate whether different training paradigms avoid the "offset" problem

## Conclusion: The Road Ahead

Our study of PPO-aligned GPT-2 reveals that current alignment techniques may be more fragile than they appear. While PPO successfully changes surface behavior, it leaves underlying capabilities largely intact—creating potential vulnerabilities that could be exploited by sophisticated adversaries.

This doesn't mean current alignment research is worthless, but it does suggest we need:

1. **More robust evaluation methods** that go beyond behavioral testing
2. **Mechanistic understanding** of how alignment techniques actually work
3. **Adversarial red-teaming** to identify potential failure modes
4. **Fundamental advances** in alignment techniques that address root causes

As systems become more powerful, understanding these fundamental limitations becomes increasingly critical. The path to truly safe and aligned AI will require not just better training techniques, but a deeper understanding of how these techniques actually change the models we're trying to align.

---

*The complete codebase and experimental details are available on [GitHub](https://github.com/surajK610/rl-gpt2-sentiment). This work was conducted in collaboaration with David Getzen.*

## Technical Appendix


**Mechanistic Interpretability:**
- **Logit Lens**: Project intermediate representations to vocabulary space
- **Value Vector Analysis**: Identify which neurons promote specific concepts
- **Activation Patching**: Modify internal activations to test causal effects

**Training Configuration:**
```python
PPOConfig(
    num_rollouts=128,
    chunk_size=128,
    ppo_epochs=4,
    init_kl_coef=0.001,
    cliprange=0.2,
    gen_kwargs=dict(max_new_tokens=40, top_k=0, top_p=1.0, do_sample=True)
)
```

**Evaluation Metrics:**
- Sentiment classification scores (DistilBERT-IMDB)
- Weight cosine similarity analysis
- Activation magnitude tracking
- Token-level negative concept detection

### Reproducibility

All experiments were conducted using the `trlx` library for RLHF training and `transformer_lens` for mechanistic interpretability analysis.