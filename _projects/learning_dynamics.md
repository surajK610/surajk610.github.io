---
layout: page
title: dynamics
description: learning dynamics in MLMs
img: assets/img/learning-dynamics.png
importance: 2
category: spring 2025
---

This is work that was completed with Jack Merullo and Michael Lepori on route to [publication](https://surajk610.github.io/assets/pdf/Dual_Process_Learning_Con.pdf). It was conduected at the Brown University Lunar Lab under supervision of Ellie Pavlick.


# Developmental Narrative of Syntax in MLMs

*Discovering the "push-down effect": how syntactic knowledge migrates from late to early layers during training, and what this reveals about in-context vs. in-weights learning strategies in masked language models*

## Where Does Grammar Live in MLMs?

When BERT processes the sentence "I will mail the letter," how does it know that "mail" is functioning as a verb rather than a noun? Previous research has shown that language models develop sophisticated syntactic understanding, but a fundamental question remained unanswered: **How do these representations develop during training?**

Our research reveals a surprising phenomenon we call the **"push-down effect"**: syntactic information initially appears in later layers of the network but gradually migrates to earlier layers as training progresses. This migration tells a deeper story about how language models balance two competing strategies for understanding language.

## Two Strategies for Understanding Language

Language models can represent syntactic information through two fundamentally different approaches:

### In-Context Learning: The Algorithmic Approach
- **Strategy**: Derive part-of-speech information from surrounding context
- **Example**: "mail" after "will" → must be a verb
- **Advantage**: Handles ambiguous words flexibly
- **Location**: Typically requires deeper network layers for contextual reasoning

### In-Weights Learning: The Memorization Approach  
- **Strategy**: Encode part-of-speech information directly in word embeddings
- **Example**: "sofa" → almost always a noun, store this in the embedding
- **Advantage**: Fast, direct lookup for unambiguous words
- **Location**: Can be accessed from shallow network layers

**The Central Question**: When and why do models choose one strategy over the other?

## Discovering the Push-Down Effect

We analyzed the training dynamics of MultiBERT models using **linear probing**—training simple classifiers to extract syntactic information from different layers at various training checkpoints.


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/learning_dynamics/probing-process.png" title="probing process" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Probing setup diagram showing how we test syntactic tasks across layer and training step axes
</div>

**Syntactic Tasks Studied**:
- **Part-of-speech tagging** (coarse and fine-grained)
- **Named entity recognition**
- **Dependency parsing**
- **Phrase boundaries** (start/end detection)
- **Parse tree structure** (depth and distance)

### The Striking Pattern

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/learning_dynamics/syntax_easy.png" title="syntax_easy" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/learning_dynamics/syntax_hard.png" title="syntax_hard" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     Probing Results measured across Layer vs. Step for MultiBERT Seed 0. Results are consistent across seeds. Top is easier syntactic tasks and bottom is more difficult tasks.
</div>

**What we observed**:
- **Early training**: Syntactic information only accessible in later layers (8-12)
- **Mid training**: Information becomes available in middle layers (4-8)
- **Late training**: Full syntactic knowledge accessible in early layers (1-3)

**Two distinct regions emerged**:
1. **Upper triangle**: High accuracy, low variance (successful syntactic extraction)
2. **Lower triangle**: Poor performance (information not yet available at that layer/time)

### Task-Specific Migration Timing

Different syntactic properties "pushed down" at different rates:

**Migration Order** (fastest to slowest):
1. **Named Entity Recognition** → Quick migration to early layers
2. **Phrase boundaries** → Moderate migration speed  
3. **Part-of-speech tagging** → Steady migration
4. **Dependency parsing** → Slower migration
5. **Parse tree depth/distance** → Remained in deeper layers

**Key Insight**: Simpler syntactic properties migrated earlier, while complex structural relationships required deeper processing throughout training.

## The Synthetic Experiment: Controlling the Variables

To understand what drives the push-down effect, we designed a controlled synthetic task that captures the essence of part-of-speech disambiguation.

### Task Design

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/learning_dynamics/synthetic_task.png" title="synthetic_task" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Synthetic task diagram showing sequence patterns
</div>

**Grammar Structure**:
- **Sequence types**: `<noun> cop <adj>` or `cop <adj> <noun>` (50% probability each)
- **Query**: Ask for specific `<noun>` or `<adj>` token
- **Challenge**: Model must determine POS of query token to generate correct pattern

**Two Learning Strategies Available**:
1. **Algorithmic**: Use position relative to "cop" (copula) to determine POS
2. **Memorization**: Encode which tokens are nouns vs. adjectives in embeddings

### Critical Variables Tested

**1. Distribution Type**: Uniform vs. Zipfian (natural language-like)
**2. Vocabulary Size**: 100 to 20,000 tokens
**3. Ambiguity Level**: 0% to 50% chance tokens can switch roles

### Breakthrough Results


<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/learning_dynamics/syntactic_zipfian.png" title="syntactic_zipfian" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="/assets/img/learning_dynamics/syntactic_uniform.png" title="syntactic_uniform" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Feature representations originate in higher layers and layer arise in lower layers in the synthetic task for Zipfian distributions, not for Uniform.
</div>

#### Distribution Drives Strategy Selection

**Zipfian Distribution** (like natural language):
- **Push-down effect observed**: Information migrated from late to early layers
- **Strategy transition**: Started with in-context learning, moved to memorization
- **Critical insight**: Long tail of rare words forced development of algorithmic strategy

**Uniform Distribution**:
- **No push-down effect**: Information appeared in early layers immediately  
- **Pure memorization**: Model relied entirely on in-weights strategy
- **No algorithmic development**: No pressure to develop contextual reasoning

#### The Role of Ambiguity


<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/learning_dynamics/zipf_amb=0.00_vs=20000_a=1.5.png" title="syntactic_zipfian" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="/assets/img/learning_dynamics/zipf_amb=0.10_vs=20000_a=1.5.png" title="syntactic_uniform" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Effect of ambiguity on learning strategies. (Left is p=0.00 ambiguous POS, Right is p=0.10 ambiguous POS).

   Validation and tail probing with differing probabilities of ambiguous POS (i.e. noun being chosen from adjective set and vice versa); tokens are chosen from a Zipf distribution with $\alpha=1.001$ and a vocab size of 10,000.
</div>

**No Ambiguity (0%)**: 
- Model could rely purely on memorization
- Switch accuracy remained low (memorization successful)

**Any Ambiguity (≥1%)**:
- **Forced algorithmic strategy**: Model had to use context when tokens could switch roles
- **Switch accuracy matched validation accuracy**: Clear evidence of in-context learning
- **Surprising finding**: Even with algorithmic strategy, POS information still stored in embeddings


### 1. Natural Language Statistics Drive Algorithmic Development

**The Zipfian Effect**: Natural language's long-tailed distribution (many rare words) forces models to develop in-context strategies because memorization alone cannot handle the full vocabulary.

**Training Progression**:
1. **Early**: Learn frequent words through memorization
2. **Middle**: Develop algorithmic strategy for rare words  
3. **Late**: "Distill" algorithmic knowledge into early layers for efficiency

### 2. Memorization and Contextualization Coexist

**False Dichotomy**: The field often frames memorization vs. generalization as competing forces, but our results show they're **complementary strategies**:

- **Memorization**: Handles frequent, unambiguous cases efficiently
- **Contextualization**: Handles rare and ambiguous cases accurately
- **Integration**: Models use both strategies simultaneously

### 3. Architecture Efficiency Through Knowledge Distillation

**The Push-Down Mechanism**: Later layers develop algorithmic strategies, then "teach" earlier layers to encode this knowledge more efficiently.

**Computational Advantage**: 
- **Deep processing**: Available when needed for complex cases
- **Shallow access**: Quick lookup for common cases
- **Best of both worlds**: Flexibility with efficiency

### The Universality Question

**Cross-Task Generalization**: The push-down effect appeared across multiple syntactic tasks, suggesting a **general principle** of neural network training rather than task-specific behavior.

**Scaling Implications**: If this pattern holds for larger models and more complex linguistic phenomena, it could inform:
- **Efficient model architecture** design
- **Training procedure** optimization
- **Interpretability methodology** development

### Developmental Probing Framework

**Innovation**: Rather than probing static, fully-trained models, we systematically tracked representational changes across:
- **12 transformer layers** 
- **20+ training checkpoints**
- **8 syntactic tasks**
- **3 model seeds**

**Reproducible Patterns**: Consistent effects across seeds and tasks provide strong evidence for the generality of our findings.

### Progress Measures for Strategy Detection

**Novel Metrics**:
- **Switch Accuracy**: Test model on swapped noun/adjective roles to detect algorithmic vs. memorization strategies
- **Tail Accuracy**: Evaluate performance on rare tokens to assess generalization
- **Unseen Token Performance**: Test completely novel tokens to isolate algorithmic capability

**Causal Understanding**: These measures allowed us to not just observe strategy changes but understand **why** they occurred.

## Future Research

These are just initial steps. Some interesting questions that would help validate this study are here.
### Scaling to Modern Models

**Open Questions**:
- Do GPT-3/4 scale models show similar push-down effects?
- How do these dynamics change with billions of parameters?
- Does the effect persist across different model architectures?

### Cross-Linguistic Generalization

**Research Directions**:
- Do morphologically rich languages show different patterns?
- How do syntactic development timelines vary across language families?
- Can we predict optimal training curricula based on linguistic properties?

### Applications to Model Development

**Practical Extensions**:
- **Early stopping criteria**: Use syntactic development as training completion signal
- **Architecture search**: Design layers to support beneficial migration patterns
- **Debugging tools**: Identify when models develop problematic learning strategies

## Conclusion

Our research reveals that language models don't simply "learn syntax"—they develop increasingly sophisticated strategies for representing and accessing syntactic information. The push-down effect demonstrates a previously unknown training dynamic where **models actively reorganize their knowledge** to balance computational efficiency with representational flexibility.

**Key Takeaways**:

1. **Syntactic knowledge migrates** from deep to shallow layers during training
2. **Natural language statistics** drive the development of algorithmic reasoning strategies  
3. **Memorization and contextualization** are complementary, not competing approaches
4. **Training dynamics** provide crucial insights beyond static model analysis

**Broader Impact**: Understanding how models develop linguistic knowledge—rather than just what knowledge they possess—opens new avenues for building more efficient, interpretable, and robust language models.

The "push-down effect" isn't just a curiosity about transformer training dynamics. It reveals fundamental principles about how neural networks can efficiently organize knowledge, suggesting that the most effective AI systems may be those that, like our models, learn to balance multiple complementary strategies for understanding their domain.

---

*This research provides a developmental perspective on language model interpretability, showing that the journey of learning is as revealing as the destination. The complete experimental framework and findings offer new tools for understanding and improving language model training.*


### Experimental Details

**Models**: MultiBERT seeds 0, 1, 2 (BERT-base architecture)
**Training Checkpoints**: 0 to 2M steps (20 checkpoints)
**Probing Setup**: Linear classifiers trained on frozen representations
**Datasets**: English UD Treebank, OntoNotes-5, Penn Treebank-3

**Synthetic Task**:
- **Architecture**: 6-layer BERT with single attention head per layer
- **Vocabulary**: 100-20,000 tokens
- **Distributions**: Uniform and Zipfian (α = 1.0-1.5)
- **Ambiguity**: 0-50% token role switching probability

**Code Availability**: Experimental framework designed for reproducibility and extension to other linguistic phenomena and model architectures.

## References
Please refer to [publication](https://surajk610.github.io/assets/pdf/Dual_Process_Learning_Con.pdf) for references.
