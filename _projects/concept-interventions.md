---
layout: page
title: concept-mod
description: concept ablating and altering algorithms in neural networks
img: assets/img/concepts.png
importance: 6
category: spring 2023
---
# Testing Concept Ablation Algorithms

*Can we surgically remove concepts from neural networks? We put three leading methods to the test on a controlled dataset to find out what really works—and what doesn't.*

## Editing Neural Network "Minds"

Imagine you've trained a neural network that's learned to associate certain gender pronouns with specific professions, or developed unwanted biases about race and criminal justice. Can you surgically remove these problematic associations without retraining the entire model? This is the promise of **concept ablation**—the ability to selectively edit what neural networks have learned.

While this capability would be transformative for AI safety and fairness, the reality is more complex. Recent methods claim to "erase" concepts from neural representations, but **do they actually work?** And more importantly, **how do they compare to each other?**

Our research tackles these questions head-on by systematically comparing three leading concept intervention methods:

- **INLP (Iterative Nullspace Projection)**: Projects away linear directions encoding unwanted concepts
- **RLACE (Relaxed Linear Adversarial Concept Erasure)**: Uses adversarial training to find optimal concept-removing projections  
- **WTMT (Gradient-Based Probe Updating)**: Directly modifies activations using gradient-based optimization

## Controlled Concept Testing

Previous work on concept ablation has primarily focused on natural language processing, where concepts like "gender" or "nationality" are inherently complex and intertwined. This makes it nearly impossible to definitively prove whether a concept has been truly removed or simply hidden. Instead, we test these methods on a **synthetic dataset with perfectly controlled, independent concepts**.

### The Dataset: Compositional Visual Concepts

We used a carefully designed dataset from "Unit Testing for Concepts in Neural Networks" with three independent visual concepts:



<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/dataset.png" title="dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Dataset examples showing the three concept dimensions
</div>


**Concept Dimensions**:
- **Layout**: horizontal, vertical, ring (3 values)  
- **Shape**: rectangle, oval, polygon (3 values)
- **Stroke**: clean, fuzzy (2 values)

**Total Classes**: 18 unique combinations (3 × 3 × 2)

**Key Advantage**: Since we constructed these concepts to be truly independent, we can definitively measure whether intervention methods successfully isolate and remove specific concepts without affecting others.

### Experimental Setup

**Base Model**: Pre-trained CLIP-ViT with frozen weights
**Fine-tuning**: Added linear heads (1-layer, 2-layer, 3-layer) for classification
**Evaluation**: 5 different random seeds, 100 epochs of training
**Target**: >95% classification accuracy before intervention

## Method 1: INLP - The Linear Projection Approach


INLP removes concepts by finding the linear subspace that best encodes the target concept, then projecting it away:

```python
# Simplified INLP procedure
def inlp_ablation(representations, concept_labels):
    # Train probe to detect concept
    probe = train_linear_probe(representations, concept_labels)
    
    # Find nullspace projection
    P = compute_nullspace_projection(probe.weights)
    
    # Project away concept direction
    ablated_reps = representations @ P
    return ablated_reps
```

**Key Assumption**: Concepts are encoded in low-dimensional linear subspaces

### Results: Limited Success


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/dataset.png" title="dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Dataset examples showing the three concept dimensions
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/all_heatmap.png" title="all heatmap" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: INLP, Middle: RLACE, Right: WTMT. Accuracy Heatmap of each probe concept and the composed downstream class after intervention sorted by the original class. This heatmap indicates that WTMT affects the original class prediction and intervened concept the most. As expected, concepts that were not intervened on maintain high accuracy (indicating potential modularity of concept representations)
</div>
**What we found**:
- **Probe accuracy dropped** for targeted concepts (good sign)
- **Downstream classification remained high** (concerning sign)
- **Uneven ablation**: Some concept values (like "oval" shapes) were harder to remove than others

**Interpretation**: INLP successfully confused probes but didn't actually prevent the network from reconstructing concept information for final predictions. The network appears to have **multiple redundant pathways** for concept representation.

**Critical Limitation**: Only removes **one rank** from the representation space. Removing more ranks caused catastrophic damage to all concepts, making the method impractical for complex concept removal.

## Method 2: RLACE - The Adversarial Game Changer


RLACE frames concept removal as a **minimax game** between a concept classifier and a projection matrix:

```python
# RLACE adversarial training loop
def rlace_ablation(X, y, num_ranks=5, iterations=10000):
    P = initialize_projection_matrix(rank=num_ranks)
    
    for i in range(iterations):
        # Train classifier on projected data
        X_proj = X @ P
        classifier = train_classifier(X_proj, y)
        
        # Update projection to maximize classifier loss
        P = update_projection(P, X, y, classifier)
        
        if convergence_criterion_met():
            break
    
    return P
```

**Key Innovation**: The adversarial training finds projections that maximally confuse concept classifiers, going beyond simple linear nullspace projection.

### Results: Much More Effective

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/out_iter_effect_ablation_accuracy_no_title.png" title="out_iter_effect_ablation_accuracy_no_title" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="/assets/img/concept_ablation/rank_effect_ablation_accuracy_no_title.png" title="rank_effect_ablation_accuracy_no_title" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Ablation Accuracy across RLACE Output Iterations (Removing 10 Ranks), Right: Ablation Accuracy across RLACE Ranks Removed (Running 10,000 Output Iters). Plot showing ablation accuracies after running RLACE with various different hyperparameters (lower accuracies correspond to better ablation).
</div>

**Dramatic Improvement**:
- **Strong concept ablation**: Probe accuracies dropped significantly  
- **Causal downstream effects**: Classification accuracy followed probe accuracy (indicating real concept removal)
- **Hyperparameter sensitivity**: Performance varied significantly with number of ranks removed and training iterations

**Key Insights**:
1. **Rank matters**: Our concepts required more than 1-dimensional removal, but too many ranks (15+) hurt performance
2. **Training instability**: More iterations sometimes led to worse performance, suggesting the minimax game doesn't always converge stably
3. **Concept-specific effects**: Different visual concepts showed different ablation difficulty

**Surprising Finding**: The default 75,000 iterations often performed worse than 10,000 iterations, indicating that **careful hyperparameter tuning is essential** for each use case.

## Method 3: WTMT - The Non-Linear Alternative


Unlike INLP and RLACE, which use linear projections, WTMT directly modifies network activations using gradient descent:

```python
# WTMT concept alteration
def wtmt_intervention(model, input_data, target_concept, new_value):
    # Forward pass to target layer
    activations = model.forward_to_layer(input_data, target_layer)
    
    # Optimize activations to change concept prediction
    for step in range(optimization_steps):
        concept_pred = concept_probe(activations)
        loss = criterion(concept_pred, new_value)
        
        # Update activations via gradient descent
        activations = activations - lr * gradient(loss, activations)
    
    # Continue forward pass with modified activations
    return model.forward_from_layer(activations, target_layer)
```

**Key Advantage**: Can potentially remove **non-linear concept representations** that projection methods miss.

### Results: Powerful but Different

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/wtmt_matching_accuracies.png" title="wtmt_matching_accuracies" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/wtmt_auxillary_accuracies.png" title="wtmt_auxillary_accuracies" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Matching Probing Arch, Right: Auxillary Probing Arch. Accuracy Barplot of Downstream Counterfactual Classes After WTMT Intervention. As expected, matching architecture updating performs better than auxillary architecture updating. Note: For the last layer, both the matching and auxillary probing arches act identical.
</div>


**Effectiveness Findings**:
- **Strong concept alteration**: Successfully changed concept predictions to target values
- **Architecture sensitivity**: Complex probes (3-layer) worked much better than linear probes
- **Layer depth matters**: Interventions on later layers were more effective
- **Concept-specific difficulty**: Stroke was hardest to alter, layout was easiest

**Matching vs. Auxiliary Architectures**:
- **Matching**: Probe attached to final layer, gradients backpropped to target layer
- **Auxiliary**: Probe attached directly to target layer
- **Result**: Matching architecture significantly outperformed auxiliary, confirming that gradients need to flow through the network's natural computational pathways

**Important Caveat**: WTMT is designed for **concept alteration** rather than ablation. It changes concepts to specific new values rather than removing them entirely.

## Comparative Analysis: What Actually Works?

### Effectiveness Ranking

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/concept_ablation/concept_accuracies.png" title="dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Various Types of Concept Accuracies (Top: WTMT, Middle: RLACE Right: INLP).
For WTMT, this figure shows the accuracy of the counterfactual class, the accuracy without considering the intervened concept, and the accuracy of the original class after intervention (this would be an incorrect class as we alter to the counterfactual class). For RLACE & INLP, this figure shows accuracy without considering the intervened concept, and the accuracy of the original class after intervention
</div>


**1. RLACE**: Most effective at true concept ablation
- Successfully damaged both probe accuracy and downstream classification
- Showed causal relationship between concept removal and task performance
- Required careful hyperparameter tuning

**2. WTMT**: Most effective at concept modification  
- Excellent at changing concepts to specific target values
- Preserved other concept representations well
- Limited to alteration rather than complete removal

**3. INLP**: Least effective overall
- Only removed surface-level concept accessibility
- Network easily reconstructed concept information
- Too aggressive removal (multiple ranks) destroyed all representations

### Key Insights About Concept Representations

**1. Redundancy is Everywhere**
All methods revealed that neural networks maintain **multiple pathways** for representing concepts. Simple probe-based ablation isn't sufficient—the network can reconstruct concept information through alternative routes.

**2. Linear vs. Non-Linear Representations**
- **INLP and RLACE** assume concepts live in linear subspaces
- **WTMT** can handle non-linear concept representations
- **Reality**: Our controlled visual concepts appeared to be largely linear, which is why RLACE succeeded

**3. The Modularity Question**
The fact that we could ablate individual concepts without completely destroying others suggests some degree of **representational modularity**. However, the varying difficulty across concept types indicates this modularity isn't perfect.

### Practical Implications

**For AI Safety Applications**:
- **RLACE** shows most promise for removing unwanted biases or associations
- **Hyperparameter sensitivity** means extensive testing is required for each use case
- **No silver bullet**: Each concept may require a different approach

**For Interpretability Research**:
- **Controlled datasets** are essential for validating concept intervention methods
- **Multiple evaluation metrics** needed: probe accuracy alone is insufficient
- **Redundant representations** make concept isolation more challenging than expected

## Why This Is Difficult

### The Superposition Problem

Our findings align with recent work in mechanistic interpretability showing that neural networks use **superposition**—representing multiple concepts in overlapping ways rather than dedicating specific neurons to specific concepts.

**Evidence from our experiments**:
- Removing one concept sometimes affected others
- Different concept values showed different ablation difficulty
- Networks could reconstruct "ablated" concepts for downstream tasks

### Real-World Complexity

While our controlled visual concepts were largely linear and separable, real-world concepts in language models are likely to be:
- **More entangled**: Syntax, semantics, and pragmatics are deeply intertwined
- **Higher dimensional**: Require more complex representations than our 3-concept system
- **Hierarchically organized**: Concepts at different levels of abstraction

**Translation gap**: Success on our controlled dataset doesn't guarantee success on complex real-world applications like removing gender bias from language models.

## Lessons Learned

**1. Method Selection Matters**
- **For concept ablation**: RLACE > INLP
- **For concept alteration**: WTMT is most powerful
- **For interpretability**: All methods provide valuable but different insights

**2. Evaluation Must Be Multi-Faceted**
- **Probe accuracy**: Measures surface-level concept accessibility
- **Downstream performance**: Reveals whether concepts are truly removed
- **Cross-concept effects**: Assesses collateral damage

**3. Hyperparameter Sensitivity Is Critical**
- Default parameters rarely work optimally
- Each concept and dataset requires careful tuning
- Training stability varies significantly across methods

## The Promise and Limitations of Concept Surgery

Our systematic comparison reveals both the **promise and limitations** of current concept ablation methods. While we can successfully remove or alter concepts in controlled settings, several challenges remain:

**What Works**:
- RLACE can effectively ablate linear concept representations
- WTMT can powerfully alter concepts to specific target values  
- Controlled evaluation reveals genuine differences between methods

**What's Still Difficult**:
- Networks maintain redundant concept representations
- Hyperparameter sensitivity requires extensive tuning
- Real-world concept entanglement remains a major challenge

**The Path Forward**: Rather than seeking a universal concept ablation solution, the field may need **specialized methods for different types of concepts and applications**. Our work provides a foundation for understanding when and how these methods work, enabling more informed choices for specific use cases.

As neural networks become more powerful and deployed in higher-stakes applications, the ability to surgically edit their learned representations becomes increasingly critical. While perfect "concept surgery" remains elusive, our research shows that meaningful progress is possible—if we're careful about methodology, evaluation, and understanding the fundamental limits of current approaches.

---

*This research was conducted as part of ongoing work on neural network interpretability and concept intervention as an extension of Charles Lovering's research on Fodor's Concept Testing in Brown University's Lunar Lab. The complete experimental results and code implementations provide a foundation for future research in this rapidly evolving field.*

## Technical Appendix

**Dataset Statistics**:
- 1,000 training examples per class (18,000 total)
- Perfect balance across all concept combinations
- Clean synthetic generation ensures concept independence

**Model Architecture**:
- Base: Pre-trained CLIP-ViT (frozen weights)
- Head: Linear, 1-layer, 2-layer, or 3-layer classification heads
- Training: 100 epochs, >95% accuracy achieved across all configurations

**Hyperparameter Grids**:
- **RLACE**: Ranks removed ∈ {1, 5, 10, 15}, Iterations ∈ {1K, 10K, 50K, 75K}
- **WTMT**: Probe complexity ∈ {linear, 1-layer, 2-layer, 3-layer}, Layers ∈ {all transformer layers}
- **INLP**: Single rank removal (multiple ranks caused catastrophic failure)